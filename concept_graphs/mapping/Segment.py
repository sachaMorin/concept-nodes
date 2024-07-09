import numpy as np
import uuid


class Segment:
    def __init__(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        semantic_ft: np.ndarray,
        score: float,
        pcd_points: np.ndarray,
        pcd_rgb: np.ndarray,
        camera_pose: np.ndarray,
    ):
        self.rgb = rgb
        self.mask = mask
        self.semantic_ft = semantic_ft
        self.score = score
        self.pcd_points = pcd_points
        self.pcd_rgb = pcd_rgb
        self.camera_pose = camera_pose
        self.id = uuid.uuid4()

    def __lt__(self, other):
        return self.score < other.score
