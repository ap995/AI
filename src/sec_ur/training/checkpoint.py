"""Checkpoint utilities for saving and loading model states."""

import json
from pathlib import Path
from typing import Dict, Optional

import torch


def save_training_state(
    path: str,
    model_state: Dict,
    optimizer_state: Dict,
    scheduler_state: Optional[Dict] = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict] = None,
) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics or {},
    }
    torch.save(state, p)
    return str(p)


def load_training_state(path: str, device: str = "cpu") -> Dict:
    return torch.load(path, map_location=device)


def save_metrics(path: str, metrics: Dict) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(metrics, f, indent=2)
    return str(p)


def list_checkpoints(directory: str) -> list:
    p = Path(directory)
    return sorted([str(f) for f in p.glob("*.pt")], reverse=True)
