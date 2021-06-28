import numpy as np
import torch
from torch.autograd.grad_mode import F

LERANING_RATE: float = 0.008
B1: float = 0.85
B2: float = 0.9999
TRAIN_MEAN: np.ndarray = np.array(
    [0.42366117675939097, 0.416991564919101, 0.3933577487428576])
TRAIN_STD: np.ndarray = np.array(
    [0.24852157528053545, 0.24272031526106133, 0.24348854052275065])
TEST_MEAN: np.ndarray = np.array(
    [0.42267442091751417, 0.41485332372142375, 0.39256912718774173])
TEST_STD: np.ndarray = np.array(
    [0.24610462213210807, 0.24191442686427228, 0.24273303151710096])
EPOCHS: int = 80
BATCH_SIZE: int = 48
MODEL_SAVE_PATH: str = 'D:\\pet_projects\\film_classifier\\model_weights'
CHECKPOINT: int = 10  # 10
TRAIN_DS_PATH: str = 'D:\\pet_projects\\film_classifier\\dataset\\train_dataset'
TEST_DS_PATH: str = 'D:\\pet_projects\\film_classifier\\dataset\\test_dataset'
GAMMA: float = 0.05  # 0.05
MAX_LR: float = 0.0099  # 0.005
TRAIN_CSV_PATH: str = 'D:\\pet_projects\\film_classifier\\dataset\\df_meta.csv'
TEST_CSV_PATH: str = 'D:\\pet_projects\\film_classifier\\dataset\\df_meta.csv'
STEP_SIZE: int = 20
DF_TYPE_DICT: bool = True
TEST_STEP: int = 10
PRETRAINED: bool = True
PRETRAINED_MODEL: str = 'resnet152'
NUM_CLASSES: int = 2
LOSS_WEIGHTS: torch.Tensor = torch.tensor(
    [0.5877783531848783, 0.4122216468151217])
