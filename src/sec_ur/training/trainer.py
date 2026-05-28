"""Training loop for SecUR model with Colab fault tolerance."""

import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class SecURTrainer:
    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = scheduler
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(**batch)

        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        return {
            "loss": outputs["loss"].item(),
            "contrastive_loss": outputs.get("contrastive_loss", torch.tensor(0.0)).item(),
            "generation_loss": outputs.get("generation_loss", torch.tensor(0.0)).item(),
        }

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ret = 0.0
        total_gen = 0.0
        n_batches = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch in progress:
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            total_ret += metrics["contrastive_loss"]
            total_gen += metrics["generation_loss"]
            n_batches += 1

            progress.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "ret": f"{metrics['contrastive_loss']:.4f}",
                "gen": f"{metrics['generation_loss']:.4f}",
            })

        return {
            "loss": total_loss / n_batches,
            "contrastive_loss": total_ret / n_batches,
            "generation_loss": total_gen / n_batches,
        }

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        self.model.eval()
        loader = loader or self.eval_loader
        if loader is None:
            return {}

        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs["loss"].item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_checkpoint("best")

        return {"eval_loss": avg_loss}

    def save_checkpoint(self, tag: str = "latest") -> str:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / f"sec_ur_{tag}.pt"
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }, path)
        return str(path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))

    def train(self, num_epochs: int = 10) -> Dict[str, list]:
        history = {"loss": [], "contrastive_loss": [], "generation_loss": [], "eval_loss": []}

        for epoch in range(num_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch()

            for k, v in train_metrics.items():
                history[k].append(v)

            if self.eval_loader:
                eval_metrics = self.evaluate()
                for k, v in eval_metrics.items():
                    history.setdefault(k, []).append(v)

            self.save_checkpoint(f"epoch_{epoch}")
            self.save_checkpoint("latest")

        return history
