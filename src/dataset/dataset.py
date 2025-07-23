import os
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGTrainDataset(Dataset):
    """
    A PyTorch Dataset for ECG .pt files described in a CSV.

    Only samples with split == "train" and label != "exclude" are included.

    CSV must have columns:
      - path       : relative path to .pt file (joined with ecg_root)
      - label      : one of {"hf", "no hf", "exclude"}
      - split      : train / test / val

    .pt files should contain a single Tensor, e.g. shape (10000, 12).
    """

    _LABEL_MAP = {
        "no_hf": 0,
        "hf": 1,
        "exclude": -100,  # Excluded samples will be ignored in training
    }

    def __init__(
        self,
        csv_path: str,
        ecg_root: str,
        val_fold: int = 0,
        mode="train",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.ecg_root = os.path.expanduser(ecg_root)
        self.transform = transform
        assert mode in ["train", "val", "test"], f"Invalid mode: {mode}"
        self.mode = mode
        assert val_fold < 5, f"val_fold must be < 5, got {val_fold}"

        # Load
        df = pd.read_csv(csv_path)
        # Keep only train split
        if mode in ["train", "val"]:
            df = df[df["split"].str.lower() == "train"].copy()
            if mode == "val":
                # For validation, filter by val_fold
                df = df[df["fold"] == val_fold].copy()
            elif mode == "train":
                # For training, filter out val_fold
                df = df[df["fold"] != val_fold].copy()
        elif mode == "test":
            df = df[df["split"].str.lower() == "test"].copy()

        # Normalize label text
        df["label_clean"] = df["label"].astype(str).str.strip().str.lower()

        # Drop excludes
        if mode in ["train", "val"]:
            df = df[df["label_clean"] != "exclude"].reset_index(drop=True)

        # Map to ints
        missing = set(df["label_clean"]) - set(self._LABEL_MAP)
        if missing:
            raise ValueError(f"Found unexpected labels: {missing}")
        df["label_idx"] = df["label_clean"].map(self._LABEL_MAP)

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        pt_path = os.path.join(self.ecg_root, row["path"])
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(f"ECG file not found: {pt_path}")

        signal = torch.load(pt_path)

        if not isinstance(signal, torch.Tensor):
            raise TypeError(f"Loaded object is not a Tensor: {pt_path}")

        if self.transform is not None:
            signal = self.transform(signal)

        label = torch.tensor(row["label_idx"], dtype=torch.float)
        return signal, label.unsqueeze(0)
