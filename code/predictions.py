from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image
from qcardia_data.pipeline.data_predictor import BasePredictor, DataPredictor
from qcardia_models.metrics import DiceMetric
from qcardia_models.models import UNet2d


class CineSegmentationPredictor(BasePredictor):
    def __init__(
        self,
        wandb_run_path: Path | None = None,
        config: dict | None = None,
        name: str | None = None,
        weights_name: str = "last_model.pt",
    ):
        if wandb_run_path is not None:
            model_weights_path = wandb_run_path / "files" / weights_name
            config_path = wandb_run_path / "files" / "config-copy.yaml"
            config = yaml.load(Path.open(config_path), Loader=yaml.FullLoader)
        super().__init__(config, name)
        self.included_class_idxs = self.config["metrics"]["dice_class_idxs"]
        self.dice_metric = DiceMetric(included_class_idxs=self.included_class_idxs)

        self.model = UNet2d(
            nr_input_channels=self.config["unet"]["nr_image_channels"],
            channels_list=self.config["unet"]["channels_list"],
            nr_output_classes=self.config["unet"]["nr_output_classes"],
            nr_output_scales=self.config["unet"]["nr_output_scales"],
        ).to(self.device)

        model_weights = torch.load(model_weights_path)
        self.model.load_state_dict(model_weights)
        self.model.eval()

        self.output_dict_3d = {
            "frame_id": [],
            "dice_mean": [],
        }
        self.output_dict_2d = {
            "slice_id": [],
            "position": [],
            "dice_mean": [],
        }
        for class_idx in self.included_class_idxs:
            self.output_dict_3d[f"dice_{class_idx}"] = []
            self.output_dict_2d[f"dice_{class_idx}"] = []

    def forward_model(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data.to(self.device))[0]

    def process_summary_3d(
        self,
        subject_data_dict: dict,
        image_key: str,  # noqa: ARG002
        label_key: str,
    ) -> None:
        # 3d level dice
        dataset = subject_data_dict["meta_dict"]["dataset"]
        subject_id = subject_data_dict["meta_dict"]["subject_id"]
        frame_nr = subject_data_dict["meta_dict"]["frame_nr"]

        dice_scores = self.dice_metric(
            outputs=subject_data_dict[self.output_key].unsqueeze(0),
            targets=subject_data_dict[label_key].unsqueeze(0),
        )

        self.output_dict_3d["frame_id"].append(
            f"{dataset}-{subject_id}-__-{frame_nr:02}",
        )
        self.output_dict_3d["dice_mean"].append(torch.mean(dice_scores).item())
        for i, class_idx in enumerate(self.included_class_idxs):
            self.output_dict_3d[f"dice_{class_idx}"].append(dice_scores[i].item())

        # 2d (slice) level dice vs pos
        nr_slices = subject_data_dict[self.output_key].shape[3]
        gt_bool = torch.where(
            torch.sum(subject_data_dict[label_key][1:, ...], dim=(0, 1, 2)),
        )[0]
        base_idx = torch.min(gt_bool).item()
        apex_idx = torch.max(gt_bool).item()
        for slice_nr in range(nr_slices):
            dice_scores = self.dice_metric(
                outputs=subject_data_dict[self.output_key][..., slice_nr].unsqueeze(0),
                targets=subject_data_dict[label_key][..., slice_nr].unsqueeze(0),
            )
            self.output_dict_2d["slice_id"].append(
                f"{dataset}-{subject_id}-{slice_nr:02}-{frame_nr:02}",
            )
            self.output_dict_2d["dice_mean"].append(torch.mean(dice_scores).item())
            self.output_dict_2d["position"].append(
                (slice_nr - base_idx) / (apex_idx - base_idx),
            )
            for i, class_idx in enumerate(self.included_class_idxs):
                self.output_dict_2d[f"dice_{class_idx}"].append(dice_scores[i].item())

    def process_summary_all(self) -> dict:
        # 3d level
        output_df_3d = pd.DataFrame(self.output_dict_3d)

        # 2d level
        output_df_2d = pd.DataFrame(self.output_dict_2d)
        plt_figures = {}
        for key in output_df_2d.columns[2:]:
            plt.figure(figsize=(16, 8), tight_layout=True)
            plt.title(f"{key}: mean dice {output_df_2d[key].mean():0.4f}")
            sns.scatterplot(x="position", y=key, data=output_df_2d, color="blue")
            sns.lineplot(x="position", y=key, data=output_df_2d, color="red")
            fig = plt.gcf()
            fig.canvas.draw()
            plt_figures[f"slice_vs_pos_{key}"] = Image.frombytes(
                "RGB",
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb(),
            )
            plt.close()

        return {
            "output_3d_df": output_df_3d,
            "output_2d_df": output_df_2d,
            "stats_3d_df": get_metrics_statistics(output_df_3d),
            "stats_2d_df": get_metrics_statistics(output_df_2d),
            "plt_figures": plt_figures,
        }


def summarize_run(wandb_run_path: Path, dataset_type: str = "valid"):
    if wandb_run_path.stem == "files":
        wandb_run_path = wandb_run_path.parent
    config_path = wandb_run_path / "files" / "config-copy.yaml"
    config = yaml.load(config_path.open(), Loader=yaml.FullLoader)

    image_key, label_key = config["dataset"]["key_pairs"][0]

    model_predictor = CineSegmentationPredictor(wandb_run_path)
    data_predictor = DataPredictor(
        config=config,
        model_predictor=model_predictor,
        results_path=wandb_run_path / "files",
        data_split_file_path=wandb_run_path / "files" / "data-split.yaml",
        test=dataset_type == "test",
    )
    return data_predictor.summarize_all(dataset_type, image_key, label_key)


def save_predictions(
    wandb_run_path: Path,
    results_path: Path,
    dataset_type: str = "valid",
):
    if wandb_run_path.stem == "files":
        wandb_run_path = wandb_run_path.parent
    config_path = wandb_run_path / "files" / "config-copy.yaml"
    config = yaml.load(Path.open(config_path), Loader=yaml.FullLoader)

    image_key, label_key = config["dataset"]["key_pairs"][0]

    model_predictor = CineSegmentationPredictor(wandb_run_path)
    data_predictor = DataPredictor(
        config,
        model_predictor,
        results_path,
        data_split_file_path=wandb_run_path / "files" / "data-split.yaml",
        test=dataset_type == "test",
    )
    data_predictor.save_predictions(
        dataset_type="valid",
        file_format="pt",
        image_key=image_key,
    )


def get_metrics_statistics(
    metric_df: pd.DataFrame,
    quantiles: Iterable[float] = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0),
) -> dict:
    output_dict = {}
    for quantile in quantiles:
        output_dict[f"q{quantile:0.2f}"] = metric_df.quantile(
            quantile,
            axis=0,
            numeric_only=True,
        )
    output_dict["mean"] = metric_df.mean(axis=0, numeric_only=True)
    output_dict["std"] = metric_df.std(axis=0, numeric_only=True)
    output_dict["skew"] = metric_df.skew(axis=0, numeric_only=True)
    output_dict["kurtosis"] = metric_df.kurt(axis=0, numeric_only=True)
    return pd.DataFrame(output_dict).T.reset_index()
