"""
A script that applies DINO pretraining on a U-Net model from the qcardia-models package
on a data loader of cardiac MR images provided through the qcardia-data package. The
configuration is specified in the config yaml file. Logs training and validation metrics
to Weights & Biases and saves the best and last model weights.

example usage:
    python train_dino.py

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
from qcardia_data import DataModule
from qcardia_data.utils import data_to_file
from qcardia_models.models.building_blocks import Encoder
from qcardia_models.utils import seed_everything
from utils import (
    DINOHead,
    DINOLoss,
    MultiCropWrapper,
    cosine_scheduler,
    ensure_dataloader_sync,
    get_params_groups,
    update_config,
)

import wandb


def main():
    # the time will be recorded in the logging.
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # The config contains all the model hyperparameters and training settings for the
    # experiment. Additionally, it contains data preprocessing and augmentation
    # settings, paths to data and results, and wandb experiment parameters. Config file
    # path can be set as system argument, or defaults to a default config file.
    config_path = Path(
        "configs/dino-config.yaml" if len(sys.argv) == 1 else sys.argv[1]
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
        # mode="offline",
    )

    # get multiple config versions for global and local crops
    config_global_blur = update_config(
        wandb.config, {"data": wandb.config["global_data_update_blur"]}
    )
    config_global_solarize = update_config(
        wandb.config, {"data": wandb.config["global_data_update_solarize"]}
    )
    config_local = update_config(
        wandb.config, {"data": wandb.config["local_data_update"]}
    )
    nr_local_crops = wandb.config["local_data_update"]["nr_sample_copies"] + 1

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

    global_data_blur = DataModule(config_global_blur)
    global_data_blur.setup()

    global_data_solarize = DataModule(config_global_solarize)
    global_data_solarize.setup()

    local_data = DataModule(config_local)
    local_data.setup()

    # Outputs the paths to reformatted and cached data and the data splits to yaml file
    generated_dict = {
        "time": time,
        "paths": global_data_blur.data_cacher.get_paths_dict(),
    }
    data_to_file(generated_dict, online_files_path / "generated.yaml")
    data_to_file(global_data_blur.data_split, online_files_path / "data-split.yaml")

    # Get the MONAI DataLoader objects for the training dataset.
    global_data_blur_train_dataloader = global_data_blur.train_dataloader()
    global_data_solarize_train_dataloader = global_data_solarize.train_dataloader()
    local_data_train_dataloader = local_data.train_dataloader()
    # global_data_blur_valid_dataloader = global_data_blur.valid_dataloader()
    # global_data_solarize_valid_dataloader = global_data_solarize.valid_dataloader()
    # local_data_valid_dataloader = local_data.valid_dataloader()

    # Set up the device, loss function, model, optimizer, learning rate scheduler, and
    # early stopper. The device is set to GPU if available, otherwise CPU is used.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        warn("No GPU available; using CPU", stacklevel=1)

    # Definition of training and model settings based on the information in config yaml
    max_epochs = wandb.config["training"]["max_nr_epochs"]
    downsample_factor = 2 ** (len(wandb.config["encoder"]["channels_list"]) - 1)
    model_kwargs = {
        "nr_local_crops": nr_local_crops,
        "global_crop_encoder_output_size": [
            int(target_size / downsample_factor)
            for target_size in wandb.config["data"]["target_size"]
        ],
        "local_crop_encoder_output_size": [
            int(target_size / downsample_factor)
            for target_size in config_local["data"]["target_size"]
        ],
    }

    student_model = MultiCropWrapper(
        Encoder(
            input_channels=wandb.config["encoder"]["nr_image_channels"],
            channels=wandb.config["encoder"]["channels_list"],
        ),
        DINOHead(
            in_dim=wandb.config["encoder"]["channels_list"][-1],
            out_dim=wandb.config["dino_head"]["out_dim"],
            use_bn=wandb.config["dino_head"]["use_bn"],
            norm_last_layer=wandb.config["dino_head"]["student_norm_last_layer"],
        ),
        **model_kwargs,
    ).to(device)

    teacher_model = MultiCropWrapper(
        Encoder(
            input_channels=wandb.config["encoder"]["nr_image_channels"],
            channels=wandb.config["encoder"]["channels_list"],
        ),
        DINOHead(
            in_dim=wandb.config["encoder"]["channels_list"][-1],
            out_dim=wandb.config["dino_head"]["out_dim"],
            use_bn=wandb.config["dino_head"]["use_bn"],
            norm_last_layer=True,  # teacher model always normalizes last layer
        ),
        **model_kwargs,
    ).to(device)

    dino_loss = DINOLoss(
        out_dim=wandb.config["dino_head"]["out_dim"],
        nr_crops=2 + nr_local_crops,
        student_temperature=wandb.config["loss"]["student_temperature"],
        teacher_temperature=wandb.config["loss"]["teacher_temperature"],
        teacher_temp_warmup_start=wandb.config["loss"]["teacher_temp_warmup_start"],
        teacher_temp_warmup_epochs=wandb.config["loss"]["teacher_temp_warmup_epochs"],
        nr_epochs=max_epochs,
        center_momentum=wandb.config["loss"]["center_momentum"],
        device=device,
    )

    optimizer = torch.optim.SGD(
        get_params_groups(student_model),
        lr=wandb.config["optimizer"]["learning_rate"],
        momentum=wandb.config["optimizer"]["momentum"],
        nesterov=wandb.config["optimizer"]["nesterov"],
        weight_decay=wandb.config["optimizer"]["weight_decay"],
    )
    # optimizer = torch.optim.AdamW(get_params_groups(student_model))

    lr_schedule = cosine_scheduler(
        wandb.config["optimizer"]["learning_rate"]
        * wandb.config["dataloader"]["train"]["batch_size"]
        / 256.0,  # linear scaling rule
        wandb.config["optimizer"]["min_learning_rate"],
        max_epochs,
        len(local_data_train_dataloader),
        warmup_epochs=wandb.config["optimizer"]["warmup_epochs"],
    )
    wd_schedule = cosine_scheduler(
        wandb.config["optimizer"]["weight_decay"],
        wandb.config["optimizer"]["weight_decay_end"],
        max_epochs,
        len(local_data_train_dataloader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(
        wandb.config["optimizer"]["momentum_teacher"],
        1.0,
        max_epochs,
        len(local_data_train_dataloader),
    )

    # get image key from  first key pair
    image_key, _ = wandb.config["dataset"]["key_pairs"][0]
    best_epoch_training_loss = 1e9
    use_amp = wandb.config["training"]["mixed_precision"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # training loop
    step = 0
    for epoch_nr in range(max_epochs):
        train_loss = 0.0

        # use manuals seeds to sync dataloaders
        torch.manual_seed(wandb.config["experiment"]["seed"] + epoch_nr)
        global_data_blur_train_iter = iter(global_data_blur_train_dataloader)
        torch.manual_seed(wandb.config["experiment"]["seed"] + epoch_nr)
        global_data_solarize_train_iter = iter(global_data_solarize_train_dataloader)
        torch.manual_seed(wandb.config["experiment"]["seed"] + epoch_nr)
        local_data_train_iter = iter(local_data_train_dataloader)
        for batch_nr in range(len(local_data_train_dataloader)):
            x_global_blur = next(global_data_blur_train_iter)
            x_global_solarize = next(global_data_solarize_train_iter)
            x_local = next(local_data_train_iter)
            ensure_dataloader_sync(
                x_global_blur["meta_dict"]["file_id"],
                x_global_solarize["meta_dict"]["file_id"],
                x_local["meta_dict"]["file_id"][::nr_local_crops],
            )  # make sure dataloaders are properly synced (set num_workers > 0)

            with torch.autocast(device.type, torch.float16, enabled=use_amp):
                imgs_global = torch.concat(
                    [x_global_blur[image_key], x_global_solarize[image_key]],
                    dim=0,
                ).to(device)
                imgs_local = x_local[image_key].to(device)

                student_outputs = student_model(imgs_global, imgs_local)
                teacher_outputs = teacher_model(imgs_global, None)

                loss = dino_loss(student_outputs, teacher_outputs, epoch_nr)

            # update weight decay and learning rate according to their schedule
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[step]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[step]

            # student update
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # backpropagate loss
            if epoch_nr < wandb.config["training"]["freeze_last_layer"]:
                for n, p in student_model.named_parameters():
                    if "last_layer" in n:
                        p.grad = None
            scaler.step(optimizer)  # update weights
            scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[step]  # momentum parameter
                for param_q, param_k in zip(
                    student_model.parameters(),
                    teacher_model.parameters(),
                    strict=False,
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # log step metrics to Weights & Biases
            train_loss += loss.item()
            step_log_dict = {
                "step": step,
                "step train loss": loss.item(),
            }
            wandb.log(step_log_dict)

            step += 1  # increment training step

        # log epoch metrics to Weights & Biases
        epoch_train_loss = train_loss / len(local_data_train_dataloader)
        epoch_log_dict = {
            "epoch": epoch_nr,
            "train loss": epoch_train_loss,
        }

        wandb.log(epoch_log_dict)

        # print epoch summary, if required
        if wandb.config["general"]["verbosity"] > 0:
            epoch_str = f"{epoch_nr + 1:0{len(str(max_epochs))}}"
            train_summary = f"train loss: {epoch_train_loss:0.4f}"
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"epoch {epoch_str} | {train_summary} | {time}")

        # save model weights
        weights_dict = {k: v.cpu() for k, v in student_model.state_dict().items()}
        torch.save(weights_dict, online_files_path / "last_model.pt")
        if (epoch_nr + 1) % 10 == 0:
            torch.save(weights_dict, online_files_path / f"model_{epoch_nr + 1}.pt")
        if epoch_train_loss < best_epoch_training_loss:
            best_epoch_training_loss = epoch_train_loss
            torch.save(weights_dict, online_files_path / "best_model.pt")
            if wandb.config["general"]["verbosity"] > 0:
                print("best model updated")


if __name__ == "__main__":  # allow multiprocessing on windows
    main()
