# pure implementation of SANs
# Skrlj, Dzeroski, Lavrac and Petkovic.

"""
The code containing neural network part, Skrlj 2019
"""
import logging
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(123321)
np.random.seed(123321)

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


class TabularDataset(Dataset):
    def __init__(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        targets: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ):  # , transform=None

        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(targets, pd.DataFrame):
            target = targets.values
        self.features = torch.as_tensor(features, dtype=torch.float).flatten(start_dim=1)
        self.targets = torch.as_tensor(targets, dtype=torch.long).flatten(start_dim=1)
        unique_per_feat = [torch.unique(self.targets[:, i]) for i in range(self.targets.size(1))]
        self.num_classes = [len(unique) for unique in unique_per_feat]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.features[index]
        if self.targets is not None:
            y = self.targets[index]
        else:
            y = None

        return x, y


class MultiHeadAttention(nn.Linear):
    def __init__(self, in_features: int, num_heads: int, bias: bool = True):
        super().__init__(in_features=in_features, out_features=in_features * num_heads, bias=bias)
        self.num_heads = num_heads

    def forward(self, x: Tensor, return_softmax=False):
        out = super().forward(x)
        out = out.view(-1, self.num_heads, self.in_features).softmax(dim=-1)
        if not return_softmax:
            out = out * x.unsqueeze(1)

        return out.max(dim=1, keepdim=False)[0]

    @property
    def mean_attention_weights(self) -> Tensor:
        weight_unrolled = self.weight.data.view(self.num_heads, self.in_channels, self.in_channels)
        activated_diagonals = weight_unrolled.diagonal(dim1=1, dim2=2).softmax(1)
        output_mean = activated_diagonals.mean(0)

        return output_mean


class SANNetwork(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes: Union[int, Sequence[int]],
        hidden_layer_size,
        dropout=0.02,
        num_heads=2,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.multi_head = MultiHeadAttention(in_channels, num_heads=num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_layer_size),
            nn.Dropout(dropout),
            nn.SELU(),
            nn.Linear(hidden_layer_size, np.sum(num_classes)),
        )

        self.in_channels = in_channels
        self.num_targets = 1 if isinstance(num_classes, int) else len(num_classes)
        self.num_classes = num_classes
        self.num_heads = num_heads

    def split_outputs(self, outputs: Tensor):
        if self.num_targets > 1:
            outputs = outputs.split(self.num_classes, dim=1)
        return outputs

    def forward(
        self, x: Tensor, split: bool = False
    ) -> Tensor:  # Union[Tuple[Tensor, ...], Tensor]:
        # attend and aggregate
        out = self.multi_head(x)
        out = self.classifier(out)
        if split:
            out = self.split_outputs(out)
        return out


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

    def fit(
        self, features: Union[np.ndarray, pd.DataFrame], labels: Union[np.ndarray, pd.DataFrame]
    ):
        train_dataset = TabularDataset(features, labels)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        stopping_iteration = 0
        current_loss = float("inf")

        self.model = SANNetwork(
            features.shape[1],
            num_classes=train_dataset.num_classes,
            hidden_layer_size=self.hidden_layer_size,
            dropout=self.dropout,
            device=self.device,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        logging.info("Number of parameters: {self.num_params}")
        logging.info("Starting training for {self.num_epochs} epochs")
        for epoch in range(self.num_epochs):
            if stopping_iteration > self.stopping_crit:
                logging.info("Stopping reached!")
                break

            self.model.train()
            losses_per_batch = []
            for i, (x, y) in enumerate(dataloader):
                features = x.to(self.device)
                labels = y.to(self.device)
                outputs = self.model.forward(x, split=False)

                if self.model.num_targets > 1:
                    loss = x.new_zeros(())
                    start_dim = 0
                    for target_idx, out_dim in enumerate(train_dataset.num_classes):
                        end_dim = start_dim + out_dim
                        loss += self.loss(outputs[:, start_dim:end_dim], y[:, target_idx])
                        start_dim = end_dim
                else:
                    loss = self.loss(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_per_batch.append(float(loss))

            mean_loss = np.mean(losses_per_batch)
            if mean_loss < current_loss:
                current_loss = mean_loss
            else:
                stopping_iteration += 1
            logging.info(f"epoch {epoch}, mean loss per batch {mean_loss}")

    def predict(self, features, return_proba=False):
        test_dataset = TabularDataset(features, None)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x, _ in test_dataset:
                x = x.float().to(self.device)
                representation = self.model.forward(x)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        if not return_proba:
            a = [np.argmax(a_) for a_ in predictions]  # assumes 0 is 0
        else:
            a = []
            for pred in predictions:
                a.append(pred[1])

        return np.array(a).flatten()

    def predict_proba(self, features):
        test_dataset = TabularDataset(features, None)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in test_dataset:
                features = features.to(self.device)
                representation = self.model.forward(features)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        a = [a_[1] for a_ in predictions]
        return np.array(a).flatten()

    @property
    def mean_attention_weights(self):
        return self.model.mean_attention_weights.detach().cpu().numpy()

    def get_mean_attention_weights(self):
        return self.model.multi_head.mean_attention_weights.detach().cpu().numpy()

    def get_instance_attention(self, instance_space):
        instance_space = torch.as_tensor(instance_space, dtype=torch.float).to(self.device)
        return self.model.multi_head(instance_space, return_softmax=True).detach().cpu().numpy()
