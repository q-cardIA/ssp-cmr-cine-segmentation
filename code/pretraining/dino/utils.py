import collections.abc
from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def update_config(config, update_dict):
    config = deepcopy(config)
    return update_nested_dict(config, update_dict)


def update_nested_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def ensure_dataloader_sync(*ids_lists):
    match_bools = [
        len({ids_list[i] for ids_list in ids_lists}) == 1
        for i in range(len(ids_lists[0]))
    ]
    if not all(match_bools):
        for i in range(len(ids_lists[0])):
            ids = [ids_list[i] for ids_list in ids_lists]
            if len(set(ids)) != 1:
                print(f"mismatch at index {i}: {ids}")
        raise RuntimeError("mismatch in file ids")


# based on https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L257
class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False),
        )
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init._no_grad_trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        nr_local_crops: int,
        global_crop_encoder_output_size: Iterable[int],
        local_crop_encoder_output_size: Iterable[int],
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.nr_local_crops = nr_local_crops
        self.global_average_pool = nn.AvgPool2d(global_crop_encoder_output_size)
        self.local_average_pool = nn.AvgPool2d(local_crop_encoder_output_size)
        self.flatten = nn.Flatten()

    def forward(self, x_global, x_local):
        global_crop_features = self.global_average_pool(self.encoder(x_global)[-1])
        features = self.flatten(global_crop_features)

        if x_local is not None:
            local_crop_features = self.local_average_pool(self.encoder(x_local)[-1])
            local_crop_features = self.flatten(local_crop_features)
            local_crop_features = torch.concat(
                [
                    local_crop_features[i :: self.nr_local_crops, ...]
                    for i in range(self.nr_local_crops)
                ],
            )

            features = torch.concat([features, local_crop_features], dim=0)
        return self.head(features)


# based on https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/main_dino.py#L363
class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        nr_crops: int,
        student_temperature: float,
        teacher_temperature: float,
        teacher_temp_warmup_start: float,
        teacher_temp_warmup_epochs: int,
        nr_epochs: int,
        device: str,
        center_momentum: float,
    ):
        super().__init__()
        self.nr_crops = nr_crops
        self.student_temperature = student_temperature
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim, device=device))
        self.teacher_temperature_schedule = torch.concat(
            (
                torch.linspace(
                    teacher_temp_warmup_start,
                    teacher_temperature,
                    teacher_temp_warmup_epochs,
                ),
                torch.ones(nr_epochs - teacher_temp_warmup_epochs)
                * teacher_temperature,
            ),
        )

    def forward(self, student_output, teacher_output, epoch_nr):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temperature
        student_out = student_out.chunk(self.nr_crops)

        # teacher centering and sharpening
        temp = self.teacher_temperature_schedule[epoch_nr]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / len(teacher_output)  # * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


# based on https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L632
def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


# based on https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L187
def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_epochs=0,
    start_warmup_value=0,
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
