from typing import List
import numpy as np
from .View import View
import open3d as o3d


class Object:
    def __init__(self, rgb: np.ndarray, mask: np.ndarray, semantic_ft: np.ndarray, score: float, pcd_points: np.ndarray,
                 pcd_rgb: np.ndarray, camera_pose: np.ndarray):
        self.views: List[View] = [View(rgb, mask, semantic_ft, score, camera_pose)]
        self.n_detections = 1
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        self.pcd.colors = o3d.utility.Vector3dVector(pcd_rgb / 255.)
        self.pcd.transform(camera_pose)
        self.centroid = np.mean(pcd_points, axis=0)

    @property
    def semantic_ft(self) -> np.ndarray:
        """Pick the representative semantic vector from the views"""
        ft = [v.semantic_ft for v in self.views]
        ft = np.stack(ft, axis=0)

        mean = np.mean(ft, axis=0)

        return mean/np.linalg.norm(mean, 2)

    def __iadd__(self, other):
        self.views.extend(other.views)
        self.n_detections += other.n_detections
        self.pcd += other.pcd
        self.pcd.voxel_down_sample(voxel_size=0.01)
        self.centroid = .9 * self.centroid + .1 * other.centroid

        return self
