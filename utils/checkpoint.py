"""
utils/checkpoint.py

Simple checkpoint save/load helpers shared by train.py and test.py.
Saves model state_dict, optimizer state_dict, epoch number, and best-metric
tracking so training can be resumed exactly where it left off.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]],
    epoch: int,
    best_metric: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(ckpt_path, map_location=device)
