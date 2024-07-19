"""
A script that trains a U-Net model from the qcardia-models package on a data loader of
cardiac MR images provided through the qcardia-data package. The configuration is
specified in the config yaml file, which can specify a path to model weights to finetune
the model. Logs training and validation metrics to Weights & Biases and saves the best
and last model weights.

example usage:
    python train_unet.py
"""

import sys
from datetime import datetime
from pathlib import Path
from shutil import copy2
from warnings import warn

import qcardia_data
import qcardia_models
import torch
import yaml
from predictions import summarize_run
from qcardia_data import DataModule
from qcardia_data.utils import data_to_file
from qcardia_models.losses import DiceCELoss, EarlyStopper, MultiScaleLoss
from qcardia_models.metrics import DiceMetric
from qcardia_models.models import UNet2d
from qcardia_models.utils import seed_everything

import wandb


def main():
    # the time will be recorded in the logging.
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # The config contains all the model hyperparameters and training settings for the
    # experiment. Additionally, it contains data preprocessing and augmentation
    # settings, paths to data and results, and wandb experiment parameters. Config file
    # path can be set as system argument, or defaults to a default config file.
    config_path = Path(
        "configs/baseline-config.yaml" if len(sys.argv) == 1 else sys.argv[1],
    )
    config = yaml.load(Path.open(config_path), Loader=yaml.FullLoader)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        warn("No GPU available; using CPU", stacklevel=1)

    # Definition of training and model settings based on the information in config yaml
    max_epochs = wandb.config["training"]["max_nr_epochs"]

    loss_function = MultiScaleLoss(
        loss_function=DiceCELoss(
            cross_entropy_loss_weight=wandb.config["loss"]["cross_entropy_loss_weight"],
            dice_loss_weight=wandb.config["loss"]["dice_loss_weight"],
            dice_classes_weights=wandb.config["loss"]["dice_classes_weights"],
        ),
    )

    dice_metric = DiceMetric(wandb.config["metrics"]["dice_class_idxs"])

    unet_model = UNet2d(
        nr_input_channels=wandb.config["unet"]["nr_image_channels"],
        channels_list=wandb.config["unet"]["channels_list"],
        nr_output_classes=wandb.config["unet"]["nr_output_classes"],
        nr_output_scales=wandb.config["unet"]["nr_output_scales"],
    ).to(device)

    if wandb.config["unet"]["weights_path"].lower() == "none":
        if wandb.config["general"]["verbosity"] > 0:
            print("no weights loaded")
        nr_epochs_frozen_encoder = 0
    else:
        weights_path = Path(wandb.config["unet"]["weights_path"])
        if not weights_path.exists():
            raise FileNotFoundError(f"weights not found at {weights_path}")
        nr_epochs_frozen_encoder = wandb.config["training"]["nr_epochs_frozen_encoder"]
        state_dict = torch.load(weights_path)

        # check how many keys match between weights and model
        unet_keys = unet_model.state_dict().keys()
        nr_matching_keys = sum([key in unet_keys for key in state_dict])
        if nr_matching_keys == 0:
            raise ValueError("No keys match between weights and model.")

        unet_model.load_state_dict(state_dict, strict=False)
        torch.save(state_dict, online_files_path / "loaded_weights.pt")
        if wandb.config["general"]["verbosity"] > 0:
            key_match_percentage_loaded = nr_matching_keys / len(state_dict) * 100.0
            key_match_percentage_target = nr_matching_keys / len(unet_keys) * 100.0
            print(f"loaded weight dict from {weights_path}", end="\n    shared keys: ")
            print(f"{key_match_percentage_loaded:0.1f}% of loaded dict", end=", ")
            print(f"{key_match_percentage_target:0.1f}% of target dict.")

        if nr_epochs_frozen_encoder > 0:
            unet_model.set_encoder_requires_grad(False)
            if wandb.config["general"]["verbosity"] > 0:
                print("encoder frozen")

    optimizer = torch.optim.SGD(
        unet_model.parameters(),
        lr=wandb.config["optimizer"]["learning_rate"],
        momentum=wandb.config["optimizer"]["momentum"],
        nesterov=wandb.config["optimizer"]["nesterov"],
        weight_decay=wandb.config["optimizer"]["weight_decay"],
    )
    learning_rate_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=wandb.config["training"]["max_nr_epochs"],
        power=wandb.config["training"]["polynomial_scheduler_power"],
    )

    if wandb.config["training"]["early_stopping"]["active"]:
        early_stopper = EarlyStopper(
            patience=wandb.config["training"]["early_stopping"]["patience"],
            min_delta=wandb.config["training"]["early_stopping"]["min_delta"],
        )

    # get image key from  first key pair
    image_key, label_key = wandb.config["dataset"]["key_pairs"][0]
    best_valid_loss = 1e9  # set to a large value to ensure first validation loss update
    nr_dice_classes = len(
        wandb.config["metrics"]["dice_class_idxs"],
    )  # nr of foreground classes

    # training loop
    for epoch_nr in range(max_epochs):
        # initialize epoch metrics
        train_loss = 0.0
        train_ce_loss = 0.0
        train_dice_loss = 0.0
        train_dice_scores = torch.zeros(
            nr_dice_classes,
            dtype=torch.float32,
            device=device,
        )

        # check if encoder must be unfrozen
        if nr_epochs_frozen_encoder > 0 and epoch_nr == nr_epochs_frozen_encoder:
            unet_model.set_encoder_requires_grad(True)
            if wandb.config["general"]["verbosity"] > 0:
                print("encoder unfrozen")

        # training
        for x in train_dataloader:
            # training step
            optimizer.zero_grad()  # reset gradients
            outputs = unet_model(x[image_key].to(device))  # forward pass
            labels = x[label_key].to(device)  # get ground truth labels

            loss, ce_loss, dice_loss = loss_function(outputs, labels)  # calculate loss
            loss.backward()  # backpropagate loss
            optimizer.step()  # update weights

            # record loss, its components, and dice metric
            train_loss += loss.item()
            train_ce_loss += ce_loss.item()
            train_dice_loss += dice_loss.item()
            train_dice_scores += dice_metric(outputs[0], labels)

        # calculate mean loss and dice metric for epoch
        train_dice_scores /= len(train_dataloader)
        train_loss /= len(train_dataloader)
        train_dice = torch.mean(train_dice_scores).item()

        # validation
        with torch.no_grad():
            unet_model.eval()  # set model to evaluation mode
            # initialize validation metrics
            valid_loss = 0.0
            valid_ce_loss = 0.0
            valid_dice_loss = 0.0
            valid_dice_scores = torch.zeros(
                nr_dice_classes,
                dtype=torch.float32,
                device=device,
            )
            for x in valid_dataloader:
                # validation step
                outputs = unet_model(x[image_key].to(device))  # forward pass
                labels = x[label_key].to(device)  # get ground truth labels
                loss, ce_loss, dice_loss = loss_function(
                    outputs,
                    labels,
                )  # calculate loss

                # record validation loss, its components, and dice metric
                valid_loss += loss.item()
                valid_ce_loss += ce_loss.item()
                valid_dice_loss += dice_loss.item()
                valid_dice_scores += dice_metric(outputs[0], labels)

            # calculate mean validation loss and dice metric for epoch
            valid_dice_scores /= len(valid_dataloader)
            valid_loss /= len(valid_dataloader)
            valid_dice = torch.mean(valid_dice_scores).item()
        unet_model.train()  # set model back to training mode

        # log metrics to Weights & Biases
        log_dict = {
            "epoch": epoch_nr,
            "train loss": train_loss,
            "train ce loss": train_ce_loss / len(train_dataloader),
            "train dice loss": train_dice_loss / len(train_dataloader),
            "train dice": train_dice,
            "valid loss": valid_loss,
            "valid ce loss": valid_ce_loss / len(valid_dataloader),
            "valid dice loss": valid_dice_loss / len(valid_dataloader),
            "valid dice": valid_dice,
        }

        # log dice scores for each class
        for i, class_idx in enumerate(wandb.config["metrics"]["dice_class_idxs"]):
            log_dict[f"train dice class {class_idx}"] = train_dice_scores[i].item()
            log_dict[f"valid dice class {class_idx}"] = valid_dice_scores[i].item()

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
            train_summary = f"train loss: {train_loss:0.4f}, dice: {train_dice:0.4f}"
            valid_summary = f"valid loss: {valid_loss:0.4f}, dice: {valid_dice:0.4f}"
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"epoch {epoch_str} | {train_summary} | {valid_summary} | {time}")

        if wandb.config["training"]["early_stopping"]["active"]:  # noqa: SIM102
            if early_stopper.early_stop(valid_loss):
                break

        # save model weights (best and final)
        weights_dict = {k: v.cpu() for k, v in unet_model.state_dict().items()}
        torch.save(weights_dict, online_files_path / "last_model.pt")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(weights_dict, online_files_path / "best_model.pt")
            if wandb.config["general"]["verbosity"] > 0:
                print("best model updated")

    # Summarize run with a full run on the validation set, evaluated in original target
    # space/size on a subject (3D) and slice (2D) level.
    summary_dict = summarize_run(online_files_path)

    # Save 3D and 2D summaries and summary statistics as wandb tables
    run.summary["summary_3d"] = wandb.Table(dataframe=summary_dict["output_3d_df"])
    run.summary["summary_2d"] = wandb.Table(dataframe=summary_dict["output_2d_df"])
    run.summary["stats_3d"] = wandb.Table(dataframe=summary_dict["stats_3d_df"])
    run.summary["stats_2d"] = wandb.Table(dataframe=summary_dict["stats_2d_df"])

    # Log figures to Weights & Biases as wandb images
    for key in summary_dict["plt_figures"]:
        run.summary[key] = wandb.Image(summary_dict["plt_figures"][key])


if __name__ == "__main__":  # allow multiprocessing on windows
    main()
