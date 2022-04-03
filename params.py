import os

SMALL_DS = True

BASE_PATH = os.getcwd()
OG_DATA_NAME = "None"
OG_DATA_PATH = os.path.join(BASE_PATH, OG_DATA_NAME)
PROCESSED_DATA_PATH = os.path.join(BASE_PATH, "out", "data")
PROCESSED_DATA_OUT_PATH = os.path.join(BASE_PATH, "out", "data")
FIGURES_PATH = os.path.join(BASE_PATH, "out", "figures")
MODEL_LOAD_PATH = os.path.join(BASE_PATH, "out", "models")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "out", "models")

if SMALL_DS:
    PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "small")
    PROCESSED_DATA_OUT_PATH = os.path.join(PROCESSED_DATA_OUT_PATH, "small")
    FIGURES_PATH = os.path.join(FIGURES_PATH, "small")
    MODEL_LOAD_PATH = os.path.join(MODEL_LOAD_PATH, "small")
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "small")

model_params = {"in0": 10, "out0": 10, "in1": 10, "out1": 10, "device": "cuda"}

run_params = {
    "lr": 1e-3,
    "start_epoch": 0,
    "end_epoch": 3,
    "loss": "mse",
    "print_per_epoch": 1,
    "save_per_epoch": 1,
    "b_size": 32,
    "save_name": "Base-1",
    "load_name": "Base-1",
    "device": "cuda",
    "cross_validation": {"n_cross_v": 5, "test_index": 4},
    "incompatible": [
        "loss",
        "model_params",
    ],  # value must be the same to load successfully
    "model_params": model_params,
}
