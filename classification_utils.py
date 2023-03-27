import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


class Folds:
    """
    Class to split the data into different folds based on groups
    """

    def __init__(
        self,
        x_data: torch.Tensor,
        y_data: torch.Tensor,
        groups: Optional[torch.Tensor] = None,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int = 42,
    ):
        """
        Class to split the data into different folds based on groups

        Parameters
        ----------
        x_data : torch.Tensor
            Features for prediction
        y_data : torch.Tensor
            Variable to predict
        groups : Optional[torch.Tensor]
            Groups to split by, default None. If None is passed, then does standard KFold,
            otherwise implements GroupShuffleSplit (if shuffle is True),
            or GroupKFold (if shuffle is False)
        n_splits : int, optional
            Number of splits / folds, by default 5
        shuffle : bool, optional
            Whether or not to shuffle the dataset, by default False
        random_state : int, optional
            Seed number, by default 42

        Raises
        ------
        ValueError
            if `n_splits` < 2
        ValueError
            if `x_data` and `y_data` do not have the same number of records
            (number of rows in `x_data` should equal the length of `y_data`)
        ValueError
            if `x_data` and `groups` do not have the same number of records
            (number of rows in `x_data` should equal the length of `groups`)
        """
        if n_splits < 2:
            raise ValueError("n_splits should be at least 2")
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                "x_data and y_data do not have compatible shapes "
                + "(need to have same number of samples)"
            )
        if groups is not None:
            if x_data.shape[0] != groups.shape[0]:
                raise ValueError(
                    "x_data and groups do not have compatible shapes "
                    + "(need to have same number of samples)"
                )
        self.x_data = x_data
        self.y_data = y_data
        self.groups = groups
        self.n_splits = n_splits
        self.shuffle = shuffle
        if self.shuffle:
            self.random_state = random_state
        else:
            self.random_state = None
        if self.groups is not None:
            if self.shuffle:
                # GroupShuffleSplit does not guarantee that every group is in a test group
                self.fold = GroupShuffleSplit(
                    n_splits=self.n_splits, random_state=self.random_state
                )
            else:
                # GroupKFold guarantees that every group is in a test group once
                self.fold = GroupKFold(n_splits=self.n_splits)
        else:
            self.fold = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        self.fold_indices = list(self.fold.split(X=x_data, groups=groups))

    def get_splits(
        self,
        fold_index: int,
        dev_size: float = 0.33,
        as_DataLoader: bool = False,
        data_loader_args: dict = {"batch_size": 64, "shuffle": True},
    ) -> Union[
        Tuple[DataLoader, DataLoader, DataLoader],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        """
        Obtains the data from a particular fold

        Parameters
        ----------
        fold_index : int
            Which fold to obtain data for
        dev_size : float, optional
            Proportion of training data to use as validation data, by default 0.33
        as_DataLoader : bool, optional
            Whether or not to return as `torch.utils.data.dataloader.DataLoader` objects
            ready to be passed into PyTorch model, by default False
        data_loader_args : _type_, optional
            Any keywords to be passed in obtaining the
            `torch.utils.data.dataloader.DataLoader` object,
            by default {"batch_size": 64, "shuffle": True}

        Returns
        -------
        - If `as_DataLoader` is True, return tuple of
        `torch.utils.data.dataloader.DataLoader` objects:
          - First element is training dataset
          - Second element is validation dataset
          - Third element is testing dataset
        - If `as_DataLoader` is False, returns tuple of `torch.Tensors`:
          - First element is features for testing dataset
          - Second element is labels for testing dataset
          - First element is features for validation dataset
          - Second element is labels for validation dataset
          - First element is features for training dataset
          - Second element is labels for training dataset

        Raises
        ------
        ValueError
            if the requested fold_index is not valid
        """
        if fold_index not in list(range(self.n_splits)):
            raise ValueError(
                f"There are {self.n_splits} folds, so "
                + f"fold_index must be in {list(range(self.n_splits))}"
            )
        # obtain train and test indices for provided fold_index
        train_index = self.fold_indices[fold_index][0]
        test_index = self.fold_indices[fold_index][1]
        # obtain a validation set from the training set
        train_index, valid_index = train_test_split(
            train_index,
            test_size=dev_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        x_train = self.x_data[train_index]
        y_train = self.y_data[train_index]
        x_valid = self.x_data[valid_index]
        y_valid = self.y_data[valid_index]
        x_test = self.x_data[test_index]
        y_test = self.y_data[test_index]

        if as_DataLoader:
            train = TensorDataset(x_train, y_train)
            valid = TensorDataset(x_valid, y_valid)
            test = TensorDataset(x_test, y_test)

            train_loader = DataLoader(dataset=train, **data_loader_args)
            valid_loader = DataLoader(dataset=valid, **data_loader_args)
            test_loader = DataLoader(dataset=test, **data_loader_args)

            return train_loader, valid_loader, test_loader
        else:
            return x_test, y_test, x_valid, y_valid, x_train, y_train


def set_seed(seed: int) -> None:
    """
    Helper function for reproducible behavior to set the seed in
    `random`, `numpy`, `torch` (if installed).

    Parameters
    ----------
    seed : int
        Seed number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
