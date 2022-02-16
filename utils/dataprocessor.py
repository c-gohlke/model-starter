import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from params import SMALL_DS


class DataProcessor:
    def __init__(
        self,
        og_data_path,
        processed_data_path,
        processed_data_out_path,
        n_cross_v,
        test_withhold,
    ):
        if (
            os.path.exists(os.path.join(processed_data_path, f"data.pt"))
            and os.path.exists(os.path.join(processed_data_path, f"y_info_{i}.pkl"))
            and os.path.exists(os.path.join(processed_data_path, f"X_info_{i}.pkl"))
        ):
            self.data = None
            print("fast load")
            self.data = torch.load(os.path.join(processed_data_path, f"data.pt"))
        else:
            print("gen data")
            with open(os.path.join(og_data_path, "train.csv")) as f:
                df = pd.read_csv(f)
                print("data loaded")
                if SMALL_DS:
                    df = df[-1000:]

            df = self.add_features(df)

            X, self.X_normalizer = self.preprocess_X(df)
            y, self.y_normalizer = self.preprocess_y(df)

            self.data = None

            if not os.path.exists(processed_data_path):
                os.makedirs(processed_data_path)

            self.data = torch.cat((X, y.reshape(-1, 1)), dim=1)

            torch.manual_seed(42)
            n = self.data.shape[0]
            idx = torch.randperm(n)

            self.data = self.data[idx]

            torch.save(self.data, os.path.join(processed_data_out_path, f"data.pt"))

        self.n_cross_v = n_cross_v
        self.set_indices = self.get_cv_indexes()
        self.test_withhold = test_withhold

    def add_features(self, df):
        # TODO
        return df

    def get_cv_indexes(self):
        np.random.seed(24)
        set_indices = [None for _ in range(self.n_cross_v)]
        n = self.data.shape[0]
        idx = np.random.permutation(n)

        for k in range(5):
            set_indices[k] = idx[
                int(k * n / self.n_cross_v) : (int((k + 1) * n / self.n_cross_v))
            ]

        return set_indices

    def get_train_data(self):
        train_indices = (
            self.set_indices[: self.test_withhold]
            + self.set_indices[self.test_withhold + 1 :]
        )
        train_indices = np.concatenate(train_indices)
        train_data = self.data[train_indices]
        return train_data.reshape(-1)

    def get_test_data(self):
        test_indices = self.set_indices[self.test_withhold]
        test_data = self.data[test_indices]
        return test_data.reshape(-1)

    def preprocess_y(self, y):
        # normalize [0, 1]
        y_normalizer = MinMaxScaler()
        y = torch.tensor(y_normalizer.fit_transform(y)).float()
        return y, y_normalizer

    def preprocess_X(self, X):
        X_normalizer = MinMaxScaler()
        X = torch.tensor(X_normalizer.fit_transform(X)).float()
        return X, X_normalizer
