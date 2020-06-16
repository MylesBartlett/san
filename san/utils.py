import random
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["TabularDataset", "random_seed"]


class TabularDataset(Dataset):
    def __init__(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        targets: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ):

        if isinstance(features, pd.DataFrame):
            features = features.values
        self.features = torch.as_tensor(features, dtype=torch.float)
        if targets is None:
            self.targets = None
            self.num_classes = None
        else:
            if isinstance(targets, pd.DataFrame):
                target = targets.values
            self.targets = torch.as_tensor(targets, dtype=torch.long).view(self.features.size(0), -1)
            unique_per_feat = [torch.unique(self.targets[:, i]) for i in range(self.targets.size(1))]
            self.num_classes = [len(unique) for unique in unique_per_feat]

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor], Tensor]:
        x = self.features[index]
        if self.targets is not None:
            y = self.targets[index]
            return x, y
        else:
            return x


def random_seed(seed_value: int, use_cuda: bool) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
