# pure implementation of SANs
# Skrlj, Dzeroski, Lavrac and Petkovic.

"""
The code containing neural network part, Skrlj 2019
"""
from typing import Optional, Sequence, Union
import torch

# from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
import pandas as pd

torch.manual_seed(123321)
np.random.seed(123321)

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


class E2EDatasetLoader(Dataset):
    def __init__(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        targets: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ):  # , transform=None
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(targets, pd.DataFrame):
            target = targets.values
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index: int):
        instance = torch.as_tensor(self.features[index, :])
        if self.targets is not None:
            target = torch.as_tensor(self.targets[index])
        else:
            target = None

        return instance, target


def to_one_hot(lbx):
    enc = OneHotEncoder(handle_unknown="ignore")
    return enc.fit_transform(lbx.reshape(-1, 1))


class SANNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes: Union[int, Sequence[int]],
        hidden_layer_size,
        dropout=0.02,
        num_heads=2,
        device="cpu",
    ):
        super(SANNetwork, self).__init__()
        self.device = device

        self.multi_head = torch.nn.ModuleList(
            [torch.nn.Linear(input_size, input_size) for _ in [1] * num_heads]
        )

        self.fc_net = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.Dropout(dropout),
            nn.SELU(),
            nn.Linear(hidden_layer_size, np.sum(num_classes)),
        )

        self.num_targets = 1 if isinstance(num_classes, int) else len(num_classes)
        self.num_classes = num_classes

    def forward_attention(self, input_space, return_softmax=False):

        attention_output_space = []
        for head in self.multi_head:
            if return_softmax:
                attention_output_space.append(head(input_space).softmax(dim=1))
            else:
                # this is critical for maintaining a connection to the input space!
                attention_output_space.append(head(input_space).softmax(dim=1) * input_space)

        # initialize a placeholder
        placeholder = torch.zeros(input_space.shape).to(self.device)

        # traverse the heads and construct the attention matrix
        for element in attention_output_space:
            placeholder = torch.max(placeholder, element)

        # normalize by the number of heads
        out = placeholder  # / self.num_heads
        return out

    def get_mean_attention_weights(self):
        activated_weight_matrices = []
        for head in self.multi_head:
            wm = head.weight.data
            diagonal_els = torch.diag(wm)
            activated_diagonal = diagonal_els.softmax(dim=0)
            activated_weight_matrices.append(activated_diagonal)
        output_mean = torch.mean(torch.stack(activated_weight_matrices, axis=0), axis=0)

        return output_mean

    def forward(self, x):

        # attend and aggregate
        out = self.forward_attention(x)
        out = self.fc_net(out)

        return out

    def get_attention(self, x):
        return self.forward_attention(x, return_softmax=True)

    def get_softmax_hadamand_layer(self):
        return self.get_mean_attention_weights()


class SAN:
    def __init__(
        self,
        batch_size=32,
        num_epochs=32,
        learning_rate=0.001,
        stopping_crit=10,
        hidden_layer_size=64,
        dropout=0.2,
    ):  # , num_head=1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = torch.nn.CrossEntropyLoss()
        self.dropout = dropout
        self.batch_size = batch_size
        self.stopping_crit = stopping_crit
        self.num_epochs = num_epochs
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.num_params = None

    # not used:
    # def init_all(self, model, init_func, *params, **kwargs):
    #     for p in model.parameters():
    #         init_func(p, *params, **kwargs)

    def fit(self, features, labels):  # , onehot=False
        unique_per_feat = [np.unique(labels[:, i]) for i in range(labels.shape[1])]
        train_dataset = E2EDatasetLoader(features, labels)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        stopping_iteration = 0
        current_loss = 10000
        self.model = SANNetwork(
            features.shape[1],
            num_classes=[len(unique) for unique in unique_per_feat],
            hidden_layer_size=self.hidden_layer_size,
            dropout=self.dropout,
            device=self.device,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        logging.info("Number of parameters {}".format(self.num_params))
        logging.info("Starting training for {} epochs".format(self.num_epochs))
        for epoch in range(self.num_epochs):
            if stopping_iteration > self.stopping_crit:
                logging.info("Stopping reached!")
                break
            losses_per_batch = []
            for i, (features, labels) in enumerate(dataloader):
                features = features.float().to(self.device)
                labels = labels.long().to(self.device)
                self.model.train()
                outputs = self.model.forward(features)
                outputs = outputs.flatten(start_dim=1)
                loss = features.new_zeros(())

                if self.model.num_targets > 1:
                    start_dim = 0
                    for target_idx, out_dim in enumerate(self.model.num_classes):
                        end_dim = start_dim + out_dim
                        loss += self.loss(outputs[:, start_dim:end_dim], labels[:, target_idx])
                        start_dim = end_dim
                else:
                    loss = self.loss(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_per_batch.append(float(loss))

            mean_loss = np.mean(losses_per_batch)
            if mean_loss < current_loss:
                current_loss = mean_loss
            else:
                stopping_iteration += 1
            logging.info("epoch {}, mean loss per batch {}".format(epoch, mean_loss))

    def predict(self, features, return_proba=False):
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        with torch.no_grad():
            for features, _ in test_dataset:
                self.model.eval()
                features = features.float().to(self.device)
                representation = self.model.forward(features)
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
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in test_dataset:
                features = features.float().to(self.device)
                representation = self.model.forward(features)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        a = [a_[1] for a_ in predictions]
        return np.array(a).flatten()

    def get_mean_attention_weights(self):
        return self.model.get_mean_attention_weights().detach().cpu().numpy()

    def get_instance_attention(self, instance_space):
        instance_space = torch.from_numpy(instance_space).float().to(self.device)
        return self.model.get_attention(instance_space).detach().cpu().numpy()
