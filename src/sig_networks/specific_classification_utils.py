from __future__ import annotations

import pickle
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit


class Folds:
    def __init__(self, num_folds=5):
        # Folders to read the data from
        # directory of the dataset (datadrive or storage)
        FOLDER_complete_dataset = "/storage/adtsakal/MoC_dataset"
        # or "five_labels"
        TYPE_of_labelling = "three_labels"
        # or "intersection" (i.e., requiring majority or perfect agreement)
        TYPE_of_agreement = "majority"
        FOLDER_dataset = (
            FOLDER_complete_dataset
            + "/"
            + TYPE_of_labelling
            + "/"
            + TYPE_of_agreement
            + "/"
        )

        # list with NUM_folds sublists, each containing the
        # paths to the corresponding fold's timelines
        self.NUM_folds = num_folds
        self.FOLDER_dataset = FOLDER_dataset
        self.FOLD_to_TIMELINE = []

        for _fld in range(self.NUM_folds):
            _tmp_fldr = self.FOLDER_dataset + str(_fld) + "/"
            self.FOLD_to_TIMELINE.append(
                [
                    _tmp_fldr + f
                    for f in listdir(_tmp_fldr)
                    if isfile(join(_tmp_fldr, f))
                ]
            )

    def get_timelines_for_fold(self, fold):
        """
        Returns lists of different fields of all timelines IN the specified fold.
        Input:
            - fold (int): the fold we want to retrieve the timelines from
        Output (lists of posts):
            - timeline_ids: one tl_id per post
            - post_ids: the post_ids
            - texts: the text of each post
            - labels: the label of each post (5 possible labels)
        """
        timelines_tsv = self.FOLD_to_TIMELINE[fold]
        timeline_ids, post_ids, texts, labels = [], [], [], []
        for tsv in timelines_tsv:
            df = pd.read_csv(tsv, sep="\t")
            if (
                "374448_217" in tsv
            ):  # manually found (post 5723227 was not incorporated for some reason)
                df = pd.read_csv(tsv, sep="\t", quotechar="'")
            pstid, txt, lbl = df.postid.values, df.content.values, df.label.values
            for i in range(len(pstid)):
                timeline_ids.append(tsv.split("/")[-1][:-4])
                post_ids.append(pstid[i])
                texts.append(str(txt[i]))
                labels.append(lbl[i])
        return timeline_ids, post_ids, texts, np.array(labels)

    def get_timelines_except_for_fold(self, fold):
        """
        Returns lists of different fields of all timelines
        EXCEPT FOR the specified fold.

        Input:
            - fold (int): the fold we want to avoid retrieving the timelines from
        Output (lists of posts):
            - timeline_ids: one tl_id per post
            - post_ids: the post_ids
            - texts: the text of each post
            - labels: the label of each post (5 possible labels)
        """
        timeline_ids, post_ids, texts, labels = [], [], [], []
        for f in range(len(self.FOLD_to_TIMELINE)):
            if f != fold:
                tlids, pstid, txt, lbl = self.get_timelines_for_fold(f)
                for i in range(len(pstid)):
                    timeline_ids.append(tlids[i])
                    post_ids.append(pstid[i])
                    texts.append(str(txt[i]))
                    labels.append(lbl[i])
        return timeline_ids, post_ids, texts, np.array(labels)

    def get_labels(self, df):
        # dictionary of labels - 3-class classification
        y_dict3 = {}
        y_dict3["0"] = 0
        y_dict3["IE"] = 1
        y_dict3["IEP"] = 1
        y_dict3["IS"] = 2
        y_dict3["ISB"] = 2

        # GET THE FLAT y LABELS
        y_data = df["label"].values
        y_data = np.array([y_dict3[xi] for xi in y_data])
        y_data = torch.from_numpy(y_data.astype(int))

        return y_data

    def get_splits(self, df, x_data, y_data, test_fold, dev_size=0.33):
        # Just getting the train/test data: timelines_ids, posts_id, texts, labels
        test_tl_ids, test_pids, test_texts, test_labels = self.get_timelines_for_fold(
            test_fold
        )
        (
            train_tl_ids,
            train_pids,
            train_texts,
            train_labels,
        ) = self.get_timelines_except_for_fold(test_fold)

        timeline_test = np.unique(test_tl_ids)
        timeline_notest = np.unique(train_tl_ids)

        df_train = df[df.timeline_id.isin(timeline_notest)].reset_index(drop=True)
        splitter_tr = GroupShuffleSplit(test_size=dev_size, random_state=123)
        split_tr = splitter_tr.split(df_train, groups=df_train["timeline_id"])
        train2_inds, valid_inds = next(split_tr)

        timeline_valid = df_train[df_train.index.isin(valid_inds)][
            "timeline_id"
        ].unique()
        timeline_train = df_train[df_train.index.isin(train2_inds)][
            "timeline_id"
        ].unique()

        x_test = x_data[(df.timeline_id.isin(timeline_test)), :]
        y_test = y_data[df.timeline_id.isin(timeline_test)]
        x_valid = x_data[df.timeline_id.isin(timeline_valid), :]
        y_valid = y_data[df.timeline_id.isin(timeline_valid)]
        x_train = x_data[df.timeline_id.isin(timeline_train), :]
        y_train = y_data[df.timeline_id.isin(timeline_train)]

        test_pids_ = torch.Tensor(
            df[(df.timeline_id.isin(timeline_test))]["postid"].tolist()
        )
        test_pids_ = test_pids_.reshape(test_pids_.shape[0], 1)

        print(
            "The size of train/valid/test timelines are: ",
            timeline_train.shape[0],
            timeline_valid.shape[0],
            timeline_test.shape[0],
        )
        print("Samples in test set: ", x_test.shape[0])

        return (
            x_test,
            y_test,
            x_valid,
            y_valid,
            x_train,
            y_train,
            test_tl_ids,
            test_pids_,
        )


def process_model_results(model_code_name, FOLDER_results):
    per_model_files = [
        f for f in listdir(FOLDER_results) if model_code_name in f if "tuning" not in f
    ]
    print("There are ", len(per_model_files), " files")
    metrics_overall = pd.DataFrame(
        0,
        index=["O", "IE", "IS", "accuracy", "macro avg", "weighted avg"],
        columns=["precision", "recall", "f1-score", "support"],
    )
    with open(FOLDER_results + per_model_files[0], "rb") as fin:
        results0 = pickle.load(fin)

    for my_ran_seed in results0["classifier_params"]["RANDOM_SEED_list"]:
        labels_final = torch.empty(0)
        predicted_final = torch.empty(0)

        seed_files = [f for f in per_model_files if (str(my_ran_seed) + "seed") in f]
        for sf in seed_files:
            with open(FOLDER_results + sf, "rb") as fin:
                results = pickle.load(fin)
                labels_results = results["labels"]
                predictions_results = results["predictions"]

            # for each seed combine fold results
            labels_final = torch.cat([labels_final, labels_results])
            predicted_final = torch.cat([predicted_final, predictions_results])

        # calculate metrics for each seed
        metrics_tab = metrics.classification_report(
            labels_final,
            predicted_final,
            target_names=["O", "IE", "IS"],
            output_dict=True,
        )
        metrics_tab = pd.DataFrame(metrics_tab).transpose()
        # combine the metrics with the rest of the
        # seeds in order to take average at the end
        metrics_overall += metrics_tab

    return metrics_overall / len(results0["classifier_params"]["RANDOM_SEED_list"])
