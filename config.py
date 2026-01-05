import os
import torch

SEED = 67

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = _get_device()

def seed_all(seed=SEED):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_all(SEED)

TRAIN_SIZE = 2000
TEST_SIZE = 2000
GRID_SIZE = 1000

FUNCTIONS = ["smooth", "piecewise", "besov"]

SCALING_DEPTH = 6
SCALING_WIDTHS = [10, 20, 40, 80, 120]
SCALING_WIDTHS2 = [8, 16, 32, 64, 128]


DEPTH_VALUES = [3, 5, 8, 12]
FIXED_PARAMS = 10000
EXP2_MAX_EPOCHS = 15000

MAX_EPOCHS = 15000
BATCH_SIZE = 1024 if device.type == "cuda" else 512

EXP1_LR = 5e-4
EXP1_WEIGHT_DECAY = 0.0
EXP1_CLIP_NORM = 1.0

EXP2_LR = 3e-4
EXP2_WEIGHT_DECAY = 0.0
EXP2_CLIP_NORM = 1.0

EXP3_FUNCTIONS = ["smooth", "piecewise_kink_poly", "besov_faber_schauder"]
EXP3_WIDTHS = [10, 40, 120]
EXP3_SEEDS = [67, 68, 69]
EXP3_NOISE_SIGMAS = [0.0, 0.01]
EXP3_GRID_SIZES = [300, 1000]
EXP3_SAMPLERS = ["random", "grid"]
EXP3_WEIGHT_DECAYS = [0.0, 1e-4]
EXP3_MAX_EPOCHS = 3000
EXP3_LEARNING_RATE = 1e-3

RESULTS_DIR = "results"
os.makedirs(os.path.join(RESULTS_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
