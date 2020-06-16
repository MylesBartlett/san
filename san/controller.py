import logging
from typing import Iterator, Sequence, Union, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .model import SANNetwork
from .utils import TabularDataset, random_seed

__all__ = ["Controller"]


class Controller:
    def __init__(
        self,
        batch_size: int = 32,
        num_epochs: int = 32,
        learning_rate: float = 0.001,
        stopping_crit: int = 10,
        hidden_layer_size: int = 64,
        num_heads: int = 1,
        dropout: float = 0.2,
        seed: int = 247,
    ):
        self.hidden_layer_size = hidden_layer_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.model: SANNetwork
        self.num_params: int

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Adam
        self.stopping_crit = stopping_crit

        logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        random_seed(seed_value=seed, use_cuda=str(self.device) != "cpu")

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.model.num_targets > 1:
            loss = logits.new_zeros(())
            for target_idx, logits_k in enumerate(self.model.split_outputs(logits)):
                loss += self.loss(logits_k, targets[:, target_idx])
        else:
            loss = self.loss(logits, targets.view(-1))
        return loss

    def fit(
        self,
        x_data: Union[np.ndarray, pd.DataFrame],
        y_data: Union[np.ndarray, pd.DataFrame],
        freeze_inds: Optional[Sequence[int]] = None,
    ):
        train_dataset = TabularDataset(x_data, y_data)
        dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0
        )
        stopping_iteration = 0
        current_loss = float("inf")

        self.model = SANNetwork(
            x_data.shape[1],
            num_classes=train_dataset.num_classes,
            hidden_layer_size=self.hidden_layer_size,
            dropout=self.dropout,
            device=self.device,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Number of parameters: {self.num_params}")
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        for epoch in range(self.num_epochs):
            if stopping_iteration > self.stopping_crit:
                self.logger.info("Stopping reached!")
                break

            self.model.train()
            sum_losses = 0.0
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model.forward(x, freeze_inds=freeze_inds)

                loss = self._compute_loss(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                sum_losses += loss.item()

            mean_loss = sum_losses / len(dataloader)
            if mean_loss < current_loss:
                current_loss = mean_loss
            else:
                stopping_iteration += 1
            self.logger.info(f"Epoch {epoch} - mean loss per batch: {mean_loss:.4f}")

    @staticmethod
    def _to_numpy(tensor: Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def predict(self, features: Union[pd.DataFrame, np.ndarray], return_prob: bool = False):
        test_dataset = TabularDataset(features)
        dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        logits_ls = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                logits_ls.append(self.model.forward(x))

        logits = self.model.split_outputs(torch.cat(logits_ls, dim=0))

        if isinstance(logits, tuple):
            for logits_k in logits:
                if return_prob:
                    yield self._to_numpy(logits_k.softmax(1))
                else:
                    yield self._to_numpy(logits_k.argmax(1))
        else:
            if return_prob:
                return self._to_numpy(logits.softmax(1))
            else:
                return self._to_numpy(logits.argmax(1))

    @property
    def mean_attention_weights(self):
        return self._to_numpy(self.model.multi_head.mean_attention_weights)

    def get_instance_attention(self, features: np.ndarray):
        features_t = torch.as_tensor(features, dtype=torch.float).to(self.device)
        return self._to_numpy(self.model.multi_head(features_t, return_softmax=True))
