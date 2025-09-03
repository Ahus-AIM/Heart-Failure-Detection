import importlib
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim.optimizer
from ray.tune import Stopper
from torch import nn
from torch.utils.data import DataLoader, Subset
from yacs.config import CfgNode as CN


@contextmanager
def log_step(name: str):
    start = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Starting: {name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] Finished: {name} ({elapsed:.1f}s)")


class EarlyStopper(Stopper):
    def __init__(self, metric: str, patience: int, delta: float = 0):
        """Stops the training if the metric does not decrease by at least delta for patience epochs."""
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.delta = delta
        self.counter: Dict[str, int] = defaultdict(lambda: 0)
        self.best_score: Dict[str, float] = defaultdict(lambda: math.inf)

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        score = result[self.metric]

        if score >= self.best_score[trial_id] - self.delta:
            self.counter[trial_id] += 1
        else:
            self.counter[trial_id] = 0

        self.best_score[trial_id] = min(self.best_score[trial_id], score)

        return self.patience <= self.counter[trial_id]

    def stop_all(self) -> bool:
        return False


def get_data_loaders(
    dataset_config: Dict[Any, Any], dataloader_config: Dict[Any, Any]
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    dataset_model = import_class_from_path(dataset_config["class_path"])
    dataset_kwargs = dataset_config["KWARGS"]

    def get_transform(config: Dict[Any, Any]) -> Any:
        if "TRANSFORM" not in config:
            return None
        transform_config = config["TRANSFORM"]
        transform_class = import_class_from_path(transform_config["class_path"])
        return transform_class(**transform_config.get("KWARGS", {}))

    transform_train = get_transform(dataset_config.get("TRAIN", {}))
    transform_val = get_transform(dataset_config.get("VAL", {}))

    # Detect k-fold config
    k_folds = dataset_kwargs.get("k_folds", 1)
    val_fold = dataset_kwargs.get("val_fold", 0)

    # This is used for bloodtest pretraining where the metadata is stored in csv files as opposed to wfdb headers.
    if dataset_config.get("VAL", {}).get("KWARGS", {}).get("split") == "val":
        train_dataset = dataset_model(
            **{**dataset_kwargs, **dataset_config["TRAIN"]["KWARGS"]}, transform=transform_train
        )
        val_dataset = dataset_model(**{**dataset_kwargs, **dataset_config["VAL"]["KWARGS"]}, transform=transform_val)
        train_dataloader = DataLoader(train_dataset, **dataloader_config)
        val_dataloader = DataLoader(val_dataset, **dataloader_config)
        return train_dataloader, val_dataloader

    # Instantiate dataset once (avoid double loading) if using k-fold
    if k_folds > 1:
        # Merge TRAIN and base KWARGS, but pass transform separately
        base_kwargs = {**dataset_kwargs, **dataset_config["TRAIN"]["KWARGS"]}
        base_kwargs.pop("transform", None)  # In case transform is also in base_kwargs

        full_dataset = dataset_model(**base_kwargs, transform=transform_train)

        # Generate indices for k-fold split
        num_samples = len(full_dataset)
        indices = np.arange(num_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_sizes = np.full(k_folds, num_samples // k_folds, dtype=int)
        fold_sizes[: num_samples % k_folds] += 1
        folds = []
        current = 0
        for fold_size in fold_sizes:
            folds.append(indices[current : current + fold_size])
            current += fold_size
        val_indices = folds[val_fold]
        train_indices = np.concatenate([folds[i] for i in range(k_folds) if i != val_fold])

        train_subset = Subset(full_dataset, train_indices)
        # For validation, use the same base dataset but with val transform (if different)
        # If transforms are different, use Subset with new dataset for val
        if transform_val is not None and transform_val != transform_train:
            val_dataset = dataset_model(**base_kwargs, transform=transform_val)
            val_subset = Subset(val_dataset, val_indices)
        else:
            val_subset = Subset(full_dataset, val_indices)

        train_dataloader = DataLoader(train_subset, **dataloader_config)
        val_dataloader = DataLoader(val_subset, **dataloader_config)
        return train_dataloader, val_dataloader

    # Fallback: regular, non-kfold loader
    train_dataset = dataset_model(**{**dataset_kwargs, **dataset_config["TRAIN"]["KWARGS"]}, transform=transform_train)
    val_dataset = dataset_model(**{**dataset_kwargs, **dataset_config["VAL"]["KWARGS"]}, transform=transform_val)

    train_dataloader = DataLoader(train_dataset, **dataloader_config)
    val_dataloader = DataLoader(val_dataset, **dataloader_config)

    return train_dataloader, val_dataloader


def import_class_from_path(path: str) -> Any:
    module = importlib.import_module(".".join(path.split(".")[:-1]))
    return getattr(module, path.split(".")[-1])


def load_model(config: CN) -> nn.Module:
    model_class = import_class_from_path(config.MODEL.class_path)
    return model_class(**config.MODEL.KWARGS)  # type: ignore


def merge_ray_config_with_config(config: CN, ray_config: CN) -> CN:
    config = deepcopy(config)
    ray_config = deepcopy(ray_config)
    model_kwargs = deepcopy(config)
    for k, v in ray_config.items():
        if k in model_kwargs:
            config.MODEL.KWARGS[k] = v
    return config


class CosineToConstantLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        eta_min_divisor: Optional[float] = None,
    ):  # noqa: D107
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_min_divisor = eta_min_divisor
        super().__init__(optimizer, -1)

    def get_lr(self) -> List[float]:
        """Retrieve the learning rate of each parameter group."""
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        def _get_lr(base_lr: float) -> float:
            min_lr = self.eta_min if self.eta_min_divisor is None else base_lr / self.eta_min_divisor
            if self._step_count - 1 >= self.T_max:
                return min_lr
            return min_lr + (base_lr - min_lr) * (1 + (math.cos(math.pi * (self._step_count - 1) / self.T_max))) / 2

        return [_get_lr(base_lr) for base_lr in self.base_lrs]
