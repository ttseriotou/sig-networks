from random import choice, randrange
from typing import List

import pandas as pd
from datasets import load_dataset

# --- XXX remove
pd.options.mode.chained_assignment = None


class loadHF:
    """Load dataset from HuggingFace"""

    def __init__(
        self, dataset_name: str = "newspop", split_name: str = "train"
    ) -> None:
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.dataset = None

    def load_preprocessed_df(self, default_preprocess: str = "newspop") -> None:
        """
        Main interface for the user
        Load and preprocess a dataset from HuggingFace
        """
        self.load_df()
        self.preprocess_df(default_preprocess=default_preprocess)

    def load_dataset(self) -> None:
        """
        Uses load_dataset() from Huggingface's 'datasets' package
        """
        self.dataset = load_dataset(self.dataset_name)

    def load_df(self) -> None:
        """
        Loads the dataset as dataframe
        """
        print(f"[INFO] load dataframe, split: {self.split_name}...")
        if self.dataset is None:
            self.load_dataset()
        self.dataset_df_all = pd.DataFrame(self.dataset[self.split_name])

    def preprocess_df(self, default_preprocess="newspop") -> None:
        """
        Preprocesses the dataframe into the appropriate form for later use
        """
        if default_preprocess == "newspop":
            self.default_preprocess_newspop()

    def default_preprocess_newspop(self) -> None:
        """
        Preprocesses dataframe for "newspop" dataset
        """
        print("[INFO] preprocess...")
        # Select columns
        use_cols = ["headline", "publish_date", "topic"]
        dataset_df = self.dataset_df_all[use_cols]
        # Simplify the dates in the publish_date column
        list_datetimes = self._list_default_datetimes()
        dataset_df["publish_date"] = [
            choice(list_datetimes) for x in range(len(dataset_df))
        ]
        # timeline and post IDs
        dataset_df["timeline_id"] = dataset_df["publish_date"].apply(
            lambda x: list_datetimes.index(x)
        )
        dataset_df["post_id"] = [randrange(10) for x in range(len(dataset_df))]
        # Convert publish_date to datetime
        dataset_df["publish_date"] = pd.to_datetime(
            dataset_df["publish_date"], format="%Y-%m-%d %H:%M:%S"
        )
        # Rename the columns
        rename_cols = {
            "headline": "content",
            "publish_date": "datetime",
            "topic": "label",
            "timeline_id": "timeline_id",
            "post_id": "post_id",
        }
        dataset_df = dataset_df.rename(columns=rename_cols)
        # Encode labels
        encode_labels = {
            "label": {"economy": 2, "obama": 1, "microsoft": 0, "palestine": 0}
        }
        self.dataset_df = dataset_df.replace(encode_labels)
        print("[INFO] preprocessed dataframe can be accessed: .dataset_df")

    def _list_default_datetimes(self) -> List[str]:
        list_datetimes = [
            "2015-01-01 00:00:00",
            "2015-01-01 00:12:00",
            "2015-01-02 00:00:00",
            "2015-01-02 00:12:00",
            "2015-01-03 00:00:00",
            "2015-01-03 00:12:00",
            "2015-01-04 00:00:00",
            "2015-01-04 00:12:00",
            "2015-01-05 00:00:00",
            "2015-01-05 00:12:00",
            "2015-01-06 00:00:00",
            "2015-01-06 00:12:00",
        ]
        return list_datetimes
