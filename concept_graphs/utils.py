import os
import torch
import numpy as np
import random
import logging
from .mapping.ObjectMap import ObjectMap
import pickle

# A logger for this file
log = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    # From wanb https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info(f"Random seed set as {seed}")


def load_map(path: str) -> ObjectMap:
    map = pickle.load(open(path, "rb"))

    for obj in map:
        obj.pcd_to_o3d()

    return map
