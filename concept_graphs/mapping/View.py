import numpy as np


class View:
    def __init__(self, rgb: np.ndarray, mask: np.ndarray, semantic_ft: np.ndarray, score: float, camera_pose: np.ndarray):
        self.rgb = rgb
        self.mask = mask
        self.semantic_ft = semantic_ft
        self.score = score
        self.camera_pose = camera_pose

    def __lt__(self, other):
        return self.score < other.score
