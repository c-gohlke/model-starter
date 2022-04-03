import torch
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt

from models.basemodel import BaseModel
from utils.dataprocessor import DataProcessor
from utils.dataset import Dataset

from params import (
    OG_DATA_PATH,
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_OUT_PATH,
    MODEL_LOAD_PATH,
    MODEL_SAVE_PATH,
    FIGURES_PATH,
)


class Model:
    def __init__(self, run_params):
        self.params = run_params
        self.device = self.params["device"]
        self.model = BaseModel(self.params["model_params"])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

        self.train_data_loaded = False
        self.test_data_loaded = False

        self.train_set = None
        self.test_sets = None

        self.dataprocessor = DataProcessor(
            OG_DATA_PATH,
            PROCESSED_DATA_PATH,
            PROCESSED_DATA_OUT_PATH,
            self.params["cross_validation"]["n_cross_v"],
            self.params["cross_validation"]["test_index"],
        )

    def evaluate(self):
        pass

    def load_model(self):
        load_path = os.path.join(
            MODEL_LOAD_PATH,
            self.params["load_name"],
            f"k-cv_{self.params['cross_validation']['test_index']}.pt",
        )
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=torch.device(self.device))
            if all(
                [
                    self.params[key] == checkpoint["params"][key]
                    for key in self.params["incompatible"]
                ]
            ):
                self.model.load_state_dict(checkpoint["model"])
                self.optim.load_state_dict(checkpoint["optimizer"])
                self.params["start_epoch"] = checkpoint["epoch"] + 1
                print(f"checkpoint loaded")
            else:
                print("params incompatible")
        else:
            print(f"checkpoint to load not found")

    def load_test(self):
        self.test_set = Dataset(self.dataprocessor.get_test_data())
        self.test_data_loaded = True

    def load_train(self):
        self.train_set = Dataset(self.dataprocessor.get_train_data())
        self.train_data_loaded = True

    def loss_fn(self, pred, target, loss_function):
        if loss_function == "mse":
            ((pred - target) ** 2) / len(pred)
        else:
            pass

    def save_model(self, epoch):
        net_params = self.params.copy()
        net_params["start_epochs"] = net_params["start_epochs"]

        dir_name = os.path.join(MODEL_SAVE_PATH, self.params["save_name"],)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optim.state_dict(),
                "params": net_params,
            },
            os.path.join(
                dir_name, f"k-cv_{self.params['cross_validation']['test_index']}.pt"
            ),
        )

    def train(self, train_ids):
        print(f"----------TRAINING {train_ids}----------")
        print(f"with params {self.params}")

        train_loss_history = []
        test_loss_history = []

        if not self.train_data_loaded:
            self.load_train()

        if not self.test_data_loaded:
            self.load_test()

        self.net.train()

        for state in self.optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.params["b_size"], pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=self.params["b_size"], pin_memory=True,
        )

        for e in range(self.params["start_epochs"], self.params["end_epochs"]):
            total_train_loss = 0
            total_test_loss = 0

            train_start = time.time()
            self.model.train()

            for idx, batch in enumerate(train_loader):
                X = batch[:, :-1, :].to(self.device)
                y = batch[:, -1].to(self.device).reshape(-1)

                self.optim.zero_grad()

                y_pred = self.model(X).reshape(-1)

                loss = self.loss_fn(y_pred, y, self.params["loss"])
                loss.backward()
                self.optim.step()

                total_train_loss = total_train_loss + loss.item()

                if idx % 1000 == 0:
                    print(
                        "\r {}/{}, {:.2f}%".format(
                            idx, len(train_loader), idx / len(train_loader) * 100
                        ),
                        end="",
                    )

            total_train_loss = total_train_loss / len(train_loader)
            train_end = time.time()

            test_start = time.time()
            with torch.no_grad():
                for idx, batch in enumerate(test_loader):
                    X = batch[:, :-1, :].to(self.device)
                    y = batch[:, -1].to(self.device).reshape(-1)

                    self.model.eval()
                    y_pred = self.model(X).reshape(-1)

                    loss = self.loss_fn(y_pred, y, self.params["loss"])
                    total_test_loss = total_test_loss + loss.item()

            total_test_loss = total_test_loss / len(test_loader)
            test_end = time.time()

            if (e + 1) % self.params["print_per_epoch"] == 0:
                print(
                    (
                        f"\rEpoch {e+1}/{self.params['end_epochs']}: {train_end-train_start:.2f}s|"
                        + f"{test_end-test_start:.2f}s|"
                        + f"Train Loss:{total_train_loss:.3e}|"
                        + f"Test Loss:"
                        + f"{total_test_loss:.3e}|"
                    )
                )

            if (e + 1) % self.params["save_per_epoch"] == 0:
                self.save_model(e)
                print("model saved")

            train_loss_history.append(total_train_loss)
            test_loss_history.append(total_test_loss)

        dir_name = os.path.join(
            FIGURES_PATH, f"k-cv_{self.params['cross_validation']['test_index']}"
        )
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        sns.lineplot(
            x=list(range(self.params["start_epochs"], self.params["end_epochs"])),
            y=train_loss_history,
        )
        sns.lineplot(
            x=list(range(self.params["start_epochs"], self.params["end_epochs"])),
            y=test_loss_history,
        )
        plt.savefig(os.path.join(dir_name, f"loss.png",))
        plt.clf()

        print("training ended")
