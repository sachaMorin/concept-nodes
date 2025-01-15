import numpy as np
import uuid


class Segment:
    def __init__(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        point_map: np.ndarray,
        semantic_ft: np.ndarray,
        score: float,
        camera_pose: np.ndarray,
    ):
        self.rgb = rgb
        self.mask = mask
        self.point_map = point_map
        self.semantic_ft = semantic_ft
        self.score = score
        self.camera_pose = camera_pose
        self.id = uuid.uuid4()

    @property
    def pcd_points(self):
        return self.point_map[self.mask == 2]

    @property
    def pcd_rgb(self):
        return self.rgb[self.mask == 2]

    def __lt__(self, other):
        return self.score < other.score
