"""Colab-optimized training loop with session checkpoint resume."""

import time
from pathlib import Path
from typing import Dict, Optional

import torch

from .trainer import SecURTrainer


class ColabTrainer(SecURTrainer):
    """
    Extends SecURTrainer with:
    - Google Drive checkpoint sync
    - Session timeout management
    - Graceful shutdown and resume
    """

    def __init__(self, *args, drive_mount_path: str = "/content/drive/MyDrive/sec_ur", **kwargs):
        super().__init__(*args, **kwargs)
        self.drive_path = Path(drive_mount_path)
        self.local_checkpoint_dir = Path(self.config.output_dir)
        self.start_time = time.time()
        self.max_session_hours = 10  # Colab free limit ~12h, be safe

    def save_to_drive(self, tag: str = "latest") -> None:
        self.drive_path.mkdir(parents=True, exist_ok=True)
        local_path = self.save_checkpoint(tag)

        dest = self.drive_path / f"sec_ur_{tag}.pt"
        with open(local_path, "rb") as src_f:
            with open(dest, "wb") as dst_f:
                dst_f.write(src_f.read())
        print(f"Checkpoint synced to Drive: {dest}")

    def load_from_drive(self, tag: str = "latest") -> bool:
        path = self.drive_path / f"sec_ur_{tag}.pt"
        if path.exists():
            self.load_checkpoint(str(path))
            print(f"Loaded checkpoint from Drive: {path}")
            return True
        return False

    def should_stop_early(self) -> bool:
        elapsed = (time.time() - self.start_time) / 3600
        return elapsed >= self.max_session_hours - 0.5  # 30min buffer

    def train_with_resume(self, num_epochs: int = 10) -> Dict[str, list]:
        self.load_from_drive("latest")

        history = {"loss": [], "contrastive_loss": [], "generation_loss": [], "eval_loss": []}

        for epoch in range(self.epoch, num_epochs):
            if self.should_stop_early():
                print(f"Colab session nearing limit. Saving checkpoint at epoch {epoch}.")
                self.save_to_drive("latest")
                break

            self.epoch = epoch
            epoch_start = time.time()

            train_metrics = self.train_epoch()
            elapsed_hours = (time.time() - epoch_start) / 3600

            for k, v in train_metrics.items():
                history[k].append(v)

            if self.eval_loader:
                eval_metrics = self.evaluate()
                for k, v in eval_metrics.items():
                    history.setdefault(k, []).append(v)

            self.save_to_drive(f"epoch_{epoch}")
            self.save_to_drive("latest")

            hr = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
            print(f"[{hr}] Epoch {epoch} complete. " +
                  f"Loss: {train_metrics['loss']:.4f}, " +
                  f"Eval: {history.get('eval_loss', ['N/A'])[-1]}")

        total_hours = (time.time() - self.start_time) / 3600
        print(f"Training finished. Total time: {total_hours:.1f}h")

        return history
