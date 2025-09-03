import argparse
import ast
import os
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model.inception_ensemble import InceptionEnsemble
from src.transform.transform import DropChannels, NanToZero, Normalize
from src.utils import log_step


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml as _yaml

    with open(path, "r", encoding="utf-8") as f:
        return _yaml.safe_load(f)


class ComposedTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


transform = ComposedTransform(
    [
        DropChannels(),
        NanToZero(),
        Normalize(),
    ]
)


def get_inception_model(weights_dir: Path) -> torch.nn.Module:
    return InceptionEnsemble(weights_dir=weights_dir)


def get_ntprobnp_threshold(age, sex):
    """
    Returns the NT-proBNP threshold (pg/mL == ng/L) based on age and sex.
    """
    if pd.isna(age) or pd.isna(sex):
        return np.nan

    sex = str(sex).lower()
    if sex not in ["m", "f"]:
        sex = "m"  # default fallback

    # age may come as "68Y" or numeric; default 68 if unknown
    if isinstance(age, str) and "Y" in age:
        try:
            age = int(age.replace("Y", ""))
        except Exception:
            age = 68
    else:
        try:
            age = int(age)
        except Exception:
            age = 68

    thresholds = {
        "f": [(44, 130), (54, 249), (64, 287), (74, 301), (200, 738)],
        "m": [(44, 86), (54, 121), (64, 210), (74, 376), (200, 486)],
    }

    for max_age, threshold in thresholds[sex]:
        if age <= max_age:
            return threshold
    return 738


def _codes_from_cell(x) -> List[str]:
    """Parse code cell which might be a strified list or NaN."""
    if not x:
        return []
    if isinstance(x, list):
        return [str(c) for c in x]
    try:
        out = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(out, (list, tuple)):
            return [str(c) for c in out]
    except Exception:
        pass
    return [str(x)]


def _any_startswith(codes: Iterable[str], prefixes: Tuple[str, ...]) -> bool:
    return any(str(c).startswith(prefixes) for c in codes)


class ECGICD10BinaryDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        base_dir: str,
        ecg_csv_path: str,
        transform: Any = None,
        code_column: str = "all_diag_all",
        include_folds: Optional[List[int]] = None,
        labs_csv_path: Optional[str] = None,
        labs_itemid: Optional[int] = 50963,  # NT-proBNP
        window_days: int = 30,
        subject_col: str = "subject_id",
        ecg_time_col: str = "ecg_time",
        codes_before_col: Optional[str] = None,
        codes_during_col: Optional[str] = None,
        codes_after_col: Optional[str] = None,
        stay_id_col: Optional[str] = None,
    ) -> None:
        self.base_dir = base_dir
        self.transform = transform
        self.window_days = window_days
        self.labs_itemid = labs_itemid
        self.subject_col = subject_col
        self.ecg_time_col = ecg_time_col
        self.code_column = code_column
        self.codes_before_col = codes_before_col
        self.codes_during_col = codes_during_col or code_column
        self.codes_after_col = codes_after_col
        self.stay_id_col = stay_id_col

        self.hf_prefixes_icd10 = ("I50", "I42", "I110")
        self.hf_prefixes = self.hf_prefixes_icd10

        self.df = pd.read_csv(ecg_csv_path, parse_dates=[self.ecg_time_col], on_bad_lines="skip")

        for col in [self.code_column, self.codes_before_col, self.codes_during_col, self.codes_after_col]:
            if col and col in self.df.columns:
                self.df[col] = self.df[col].apply(_codes_from_cell)

        if include_folds is not None and "fold" in self.df.columns:
            self.df = self.df[self.df["fold"].isin(include_folds)].reset_index(drop=True)

        self.patient_all_codes: Dict[int, List[str]] = {}
        if self.subject_col not in self.df.columns:
            raise ValueError(f"Missing subject column '{self.subject_col}' in ECG CSV.")

        for sid, grp in self.df.groupby(self.subject_col, dropna=True):
            all_codes: set = set()
            for _, r in grp.iterrows():
                codes = []
                if self.code_column in r and isinstance(r[self.code_column], list):
                    codes = r[self.code_column]
                elif self.codes_during_col in r and isinstance(r[self.codes_during_col], list):
                    codes = r[self.codes_during_col]
                all_codes.update(str(c) for c in codes if pd.notna(c))
            self.patient_all_codes[int(sid)] = sorted(all_codes)

        self.labs_by_subject: Dict[int, pd.DataFrame] = {}
        if labs_csv_path is not None and os.path.exists(labs_csv_path):
            labs = pd.read_csv(labs_csv_path, parse_dates=["charttime"], on_bad_lines="skip")
            if self.labs_itemid is not None and "itemid" in labs.columns:
                labs = labs[labs["itemid"] == self.labs_itemid]

            needed = {self.subject_col, "charttime", "valuenum"}
            missing = needed - set(labs.columns)
            if missing:
                raise ValueError(f"labs_csv is missing required columns: {missing}")

            labs = labs.sort_values([self.subject_col, "charttime"]).reset_index(drop=True)
            for sid, df_sub in labs.groupby(self.subject_col, sort=False):
                self.labs_by_subject[int(sid)] = df_sub.reset_index(drop=True)

        self.df = self.df.groupby(self.subject_col, as_index=False).first()

    def __len__(self) -> int:
        return len(self.df)

    def _max_ntprobnp_ever(self, subject_id: int, ecg_time: pd.Timestamp) -> float:
        """Return max NT-proBNP ever"""
        if subject_id not in self.labs_by_subject:
            return float("nan")
        df_sub = self.labs_by_subject[subject_id]
        vals = df_sub["valuenum"]
        return float(vals.max()) if not vals.empty else float("nan")

    def _max_ntprobnp_pm30d(self, subject_id: int, ecg_time: pd.Timestamp) -> float:
        """Return max NT-proBNP within ±window_days for this subject; NaN if none."""
        if subject_id not in self.labs_by_subject or pd.isna(ecg_time):
            return float("nan")
        df_sub = self.labs_by_subject[subject_id]
        start = ecg_time - pd.Timedelta(days=self.window_days)
        end = ecg_time + pd.Timedelta(days=self.window_days)
        m = (df_sub["charttime"] >= start) & (df_sub["charttime"] <= end)
        vals = df_sub.loc[m, "valuenum"]
        return float(vals.max()) if not vals.empty else float("nan")

    def _has_hf(self, codes: Iterable[str]) -> bool:
        return _any_startswith(codes, self.hf_prefixes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        wf_path = os.path.join(self.base_dir, row["file_name"])
        record = wfdb.rdrecord(wf_path)
        signal = torch.tensor(record.p_signal, dtype=torch.float32).T
        if self.transform:
            signal = self.transform(signal)

        subject_id = int(row[self.subject_col]) if pd.notnull(row[self.subject_col]) else None
        ecg_time = row[self.ecg_time_col]

        codes_current = (
            row[self.codes_during_col]
            if self.codes_during_col in row and isinstance(row[self.codes_during_col], list)
            else []
        )
        has_hf_current = self._has_hf(codes_current)

        codes_all_patient = self.patient_all_codes.get(subject_id, []) if subject_id is not None else []
        has_hf_ever = self._has_hf(codes_all_patient)

        ntprobnp = self._max_ntprobnp_pm30d(subject_id, ecg_time) if subject_id is not None else float("nan")
        ntprobnp_ever = self._max_ntprobnp_ever(subject_id, ecg_time) if subject_id is not None else float("nan")
        ntprobnp = 0.0 if np.isnan(ntprobnp) else ntprobnp

        # -------- Build the 3-element target --------
        diag_present_current = 1.0 if has_hf_current else 0.0

        thr = get_ntprobnp_threshold(row.get("age", np.nan), row.get("gender", np.nan))
        if np.isnan(ntprobnp) or np.isnan(thr):
            middle = -1.0
        else:
            is_high = ntprobnp >= thr
            if (not has_hf_ever) and (not is_high):
                middle = 0.0  # no HF ever & low NT-proBNP
            elif has_hf_ever and is_high:
                middle = 1.0  # HF ever & high NT-proBNP
            else:
                middle = -1.0  # discordant

        if has_hf_current and (ntprobnp > 1000):
            last = 1.0  # HF during current stay & high NT-proBNP
        elif (not (has_hf_current or has_hf_ever)) and (ntprobnp_ever < 125):
            last = 0.0  # no HF ever/current & low/NaN NT-proBNP
        else:
            last = -1

        target = torch.tensor([diag_present_current, middle, last], dtype=torch.float32)

        return signal, target, wf_path


@torch.no_grad()
def run_inference(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    paths, probs, labels = [], [], []
    for x, y, path in tqdm(loader, total=len(loader)):
        x = x.to(device)
        with torch.amp.autocast(device.type):
            logits = model(x)  # (B, 1) or (B,)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
        paths.extend(path)
    return np.concatenate(probs), np.concatenate(labels), paths


def extract_predictions(y_true: np.ndarray, y_score: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    y_true = np.asarray(y_true)  # (N, 3)
    y_score = np.mean(np.asarray(y_score), axis=1)  # (N, 1)
    scores = []
    targets = []
    for i in range(y_true.shape[1]):
        mask = y_true[:, i] != -1
        targets.append(y_true[:, i][mask])
        scores.append(y_score[mask])
    return scores, targets


def auc_ci(y_true, y_score, alpha=0.95):
    """Compute AUC and CI using Hanley & McNeil (1982)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    m = ~np.isnan(y_true) & ~np.isnan(y_score)
    y_true, y_score = y_true[m], y_score[m]
    u = np.unique(y_true)
    if set(u) == {-1, 1}:
        y_true = (y_true == 1).astype(int)
    elif set(u) <= {0, 1}:
        y_true = y_true.astype(int)
    else:
        raise ValueError(f"y_true must be in {0, 1} or {-1, 1}, got {u}")
    n1, n0 = (y_true == 1).sum(), (y_true == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, np.nan, len(y_true)
    auc = roc_auc_score(y_true, y_score)
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    se = sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
    z = norm.ppf(1 - (1 - alpha) / 2)
    return auc, max(0, auc - z * se), min(1, auc + z * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="YAML config file")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    out_dir = Path(cfg.get("inference", {}).get("output_dir", "./sandbox/output"))

    device = torch.device(cfg.get("model", {}).get("device", "cpu"))

    model_cfg = cfg.get("model", {})
    weights_dir = Path(model_cfg["weights_dir"])
    model = get_inception_model(weights_dir).to(device)
    model.eval()

    paths_cfg = cfg.get("paths", {})
    inference_cfg = cfg.get("inference", {})

    with log_step("Initializing dataset"):
        dset = ECGICD10BinaryDataset(
            base_dir=paths_cfg["base_dir"],
            ecg_csv_path=paths_cfg["ecg_csv_path"],
            transform=transform,
            code_column="all_diag_all",
            labs_csv_path=paths_cfg["labs_csv_path"],
            labs_itemid=50963,  # BNP
            window_days=30,
        )
        loader = DataLoader(
            dset,
            batch_size=inference_cfg["batch_size"],
            shuffle=False,
            num_workers=inference_cfg["workers"],
            pin_memory=(device.type == "cuda"),
        )

    with log_step("Running inference"):
        probs, labels, paths = run_inference(model, loader, device)

    out_pred = out_dir / "predictions.csv"
    df = pd.DataFrame(
        {
            "path": paths,
            **{f"prob_{i+1}": np.squeeze(probs[:, i]) for i in range(probs.shape[1])},
            **{f"label_{i+1}": np.squeeze(labels[:, i]) for i in range(labels.shape[1])},
        }
    )
    df.to_csv(out_pred, index=False)

    scores, targets = extract_predictions(labels, probs)
    for i in range(len(scores)):
        auc, lower, upper = auc_ci(targets[i], scores[i])
        print(f"Target {i} AUC: {auc:.3f} (95% CI: {lower:.3f}-{upper:.3f}, n={len(scores[i])})")


if __name__ == "__main__":
    main()
