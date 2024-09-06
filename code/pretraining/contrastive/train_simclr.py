"""
A script that applies SimCLR pretraining on a U-Net model from the qcardia-models
package on a data loader of cardiac MR images provided through the qcardia-data package.
The configuration is specified in the config yaml file. Logs training and validation
metrics to Weights & Biases and saves the best and last model weights.

example usage:
    python train_simclr.py

"""

import sys
import warnings
from datetime import datetime
from pathlib import Path
from shutil import copy2

import qcardia_data
import qcardia_models
import torch
import yaml
from qcardia_data import DataModule
from qcardia_data.utils import data_to_file
from qcardia_models.losses import EarlyStopper, NTXentLoss
from qcardia_models.models import EncoderMLP2d
from qcardia_models.utils import seed_everything

import wandb


def main():
    # the time will be recorded in the logging.
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # The config contains all the model hyperparameters and training settings for the
    # experiment. Additionally, it contains data preprocessing and augmentation
    # settings, paths to data and results, and wandb experiment parameters. Config file
    # path can be set as system argument, or defaults to a default config file.
    default_config_path = "configs/simclr-config.yaml"
    config_path = Path(default_config_path if len(sys.argv) == 1 else sys.argv[1])
    config = yaml.load(config_path.open(), Loader=yaml.FullLoader)

    seed_everything(config["experiment"]["seed"])

    # Initialize a Weights & Biases run with the specified project, name, configuration,
    # and code saving settings. Sets the run mode with options "online", "offline" or
    # "disabled" to prevent logging during model training.
    run = wandb.init(
        project=config["experiment"]["project"],
        name=config["experiment"]["name"],
        id=f'{wandb.util.generate_id()}_{config["experiment"]["name"]}',
        config=config,
        save_code=True,
        mode="offline",
    )

    # Get the path to the directory where the Weights & Biases run files are stored.
    online_files_path = Path(run.dir)
    print(f"online_files_path: {online_files_path}")

    # Save copy of the config file to the Weights & Biases run directory. This preserves
    # formatting/comments and allows for easy access to the config used for the run.
    config_copy_path = online_files_path / "config-copy.yaml"
    copy2(config_path, config_copy_path)

    # Log the code for the specified packages to the Weights & Biases run. This allows
    # for easy tracking of the package versions used during model training.
    logged_packages = [qcardia_data, qcardia_models]
    for package in logged_packages:
        package_path = Path(package.__file__).parent
        package_name = package.__name__
        run.log_code(package_path, name=package_name)

    # Initialises the qcardia-data DataModule with the configuration specified in the
    # config yaml file. The DataModule handles the data loading, preprocessing, and
    # augmentation. The unique_setup method is used to cache the dataset, if a
    # previously cached verion is not already available. The setup method is used
    # to build resampling transforms including augmentations and intensity pre-processing.
    # It also builds the dataset and splits to be used by the dataloaders.
    data = DataModule(wandb.config)
    data.setup()

    # Outputs the paths to reformatted and cached data and the data splits to yaml file
    generated_dict = {
        "time": time,
        "paths": data.data_cacher.get_paths_dict(),
    }
    data_to_file(generated_dict, online_files_path / "generated.yaml")
    data_to_file(data.data_split, online_files_path / "data-split.yaml")

    # Get the MONAI DataLoader objects for the training and validation datasets.
    train_dataloader = data.train_dataloader()
    valid_dataloader = data.valid_dataloader()

    # Set up the device, loss function, model, optimizer, learning rate scheduler, and
    # early stopper. The device is set to GPU if available, otherwise CPU is used.
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Windows/Linux
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # MacOS
    else:
        device = torch.device("cpu")
        warnings.warn("No GPU available; using CPU", stacklevel=1)

    # Definition of training and model settings based on the information in config yaml
    max_epochs = wandb.config["training"]["max_nr_epochs"]

    ntxent_threshold = wandb.config["loss"]["ntxent_supervised_threshold"]
    loss_function = NTXentLoss(
        temperature=wandb.config["loss"]["ntxent_temperature"],
        threshold=None if ntxent_threshold.lower() == "none" else ntxent_threshold,
        cyclic_relative_labels=wandb.config["loss"]["ntxent_supervised_cyclic_labels"],
    )
    encoder_model = EncoderMLP2d(
        nr_input_channels=wandb.config["encoder"]["nr_image_channels"],
        encoder_channels_list=wandb.config["encoder"]["channels_list"],
        mlp_channels_list=wandb.config["mlp"]["channels_list"],
    ).to(device)

    optimizer = torch.optim.SGD(
        encoder_model.parameters(),
        lr=wandb.config["optimizer"]["learning_rate"],
        momentum=wandb.config["optimizer"]["momentum"],
        nesterov=wandb.config["optimizer"]["nesterov"],
        weight_decay=wandb.config["optimizer"]["weight_decay"],
    )
    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=wandb.config["training"]["cos_min_learning_rate"],
    )

    if wandb.config["training"]["early_stopping"]["active"]:
        early_stopper = EarlyStopper(
            patience=wandb.config["training"]["early_stopping"]["patience"],
            min_delta=wandb.config["training"]["early_stopping"]["min_delta"],
        )

    # get image key from  first key pair
    image_key, _ = wandb.config["dataset"]["key_pairs"][0]
    best_valid_loss = 1e9  # set to a large value to ensure first validation loss update
    use_amp = wandb.config["training"]["mixed_precision"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # training loop
    for epoch_nr in range(max_epochs):
        # initialize epoch metric
        train_loss = 0.0
        for x in train_dataloader:
            # training step
            optimizer.zero_grad()  # reset gradients

            # mixed precision
            with torch.autocast(device.type, torch.float16, enabled=use_amp):
                outputs = encoder_model(x[image_key].to(device))  # forward pass
                loss = loss_function(outputs)  # calculate loss
            scaler.scale(loss).backward()  # backpropagate loss
            scaler.step(optimizer)  # update weights
            scaler.update()

            # record loss
            train_loss += loss.item()

        # calculate mean loss for epoch
        train_loss /= len(train_dataloader)

        # validation step
        with torch.no_grad():
            encoder_model.eval()  # set model to evaluation mode
            # initialize validation metric
            valid_loss = 0.0
            for x in valid_dataloader:
                outputs = encoder_model(x[image_key].to(device))  # forward pass
                loss = loss_function(outputs)  # calculate loss

                # record validation loss
                valid_loss += loss.item()

            # calculate mean validation loss for epoch
            valid_loss /= len(valid_dataloader)
        encoder_model.train()  # set model back to training mode

        # log metrics to Weights & Biases
        log_dict = {
            "epoch": epoch_nr,
            "train loss": train_loss,
            "valid loss": valid_loss,
        }

        # update learning rate based on scheduler and log to Weights & Biases
        learning_rates = learning_rate_scheduler.get_last_lr()
        if len(learning_rates) == 1:
            log_dict["learning rate"] = learning_rates[0]
        else:
            for i, learning_rate in enumerate(learning_rates):
                log_dict[f"learning rate (group {i})"] = learning_rate
        learning_rate_scheduler.step()
        wandb.log(log_dict)

        # print epoch summary, if required
        if wandb.config["general"]["verbosity"] > 0:
            epoch_str = f"{epoch_nr + 1:0{len(str(max_epochs))}}"
            train_summary = f"train loss: {train_loss:0.4f}"
            valid_summary = f"valid loss: {valid_loss:0.4f}"
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"epoch {epoch_str} | {train_summary} | {valid_summary} | {time}")

        if wandb.config["training"]["early_stopping"]["active"]:  # noqa: SIM102
            if early_stopper.early_stop(valid_loss):
                break

        # save model weights (best and final)
        weights_dict = {k: v.cpu() for k, v in encoder_model.state_dict().items()}
        torch.save(weights_dict, online_files_path / "last_model.pt")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(weights_dict, online_files_path / "best_model.pt")
            if wandb.config["general"]["verbosity"] > 0:
                print("best model updated")


if __name__ == "__main__":  # allow multiprocessing on windows
    main()
