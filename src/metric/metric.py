from typing import List, Tuple, Union

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

TensorOrList = Union[torch.Tensor, List[torch.Tensor]]


def fix_input(logits_list: TensorOrList, labels_list: TensorOrList) -> Tuple[torch.Tensor, torch.Tensor]:
    if (type(logits_list) is list) and (type(labels_list) is list):
        logits = torch.cat(logits_list, dim=0).squeeze().float()
        labels = torch.cat(labels_list, dim=0).squeeze().float()
    elif (type(logits_list) is torch.Tensor) and (type(labels_list) is torch.Tensor):
        logits = logits_list.squeeze().float()
        labels = labels_list.squeeze().float()
    else:
        raise ValueError("logits and labels must be either list of tensors or tensors")

    if labels.dim() != 1 or logits.dim() != 1:
        raise ValueError("labels and logits must be 1D tensors")
    if labels.shape != logits.shape:
        raise ValueError("labels and logits must have the same shape")
    return logits, labels


def _flatten_logits_labels(
    logits_list: TensorOrList, labels_list: TensorOrList, ignore_index: int = -100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    - logits_list: either a single tensor of shape (B, C, T) or a list of such tensors
    - labels_list: either a single tensor of shape (B, T) or a list of such tensors
    Returns:
      logits_flat: (N, C) tensor
      labels_flat: (N,) tensor
    where N = number of valid (non-ignore) entries across all batches/tests.
    """
    # concat if needed
    if isinstance(logits_list, list):
        logits = torch.cat(logits_list, dim=0)
    else:
        logits = logits_list
    if isinstance(labels_list, list):
        labels = torch.cat(labels_list, dim=0)
    else:
        labels = labels_list

    # sanity
    if logits.dim() != 3:
        raise ValueError(f"logits must be 3D (B,C,T), got {logits.shape}")
    if labels.dim() != 2:
        raise ValueError(f"labels must be 2D (B,T), got {labels.shape}")
    if logits.shape[0] != labels.shape[0] or logits.shape[2] != labels.shape[1]:
        raise ValueError("Batch and test dimensions of logits and labels must match")

    B, C, T = logits.shape
    # Mask out ignore_index
    mask = labels != ignore_index  # shape (B,T)
    # Flatten
    logits_flat = logits.permute(0, 2, 1)[mask]  # (N, C)
    labels_flat = labels[mask]  # (N,)
    return logits_flat, labels_flat


def aucroc(logits_list: List[torch.Tensor] | torch.Tensor, labels_list: List[torch.Tensor] | torch.Tensor) -> float:
    logits, labels = fix_input(logits_list, labels_list)
    roc: float = float(roc_auc_score(labels.numpy(), logits.numpy()))
    return roc


def aucprc(logits_list: List[torch.Tensor] | torch.Tensor, labels_list: List[torch.Tensor] | torch.Tensor) -> float:
    logits, labels = fix_input(logits_list, labels_list)
    prc: float = float(average_precision_score(labels.numpy(), logits.numpy()))
    return prc
