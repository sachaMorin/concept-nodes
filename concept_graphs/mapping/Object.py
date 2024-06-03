from typing import List
import numpy as np
from .View import View
from .ViewHeap import ViewHeap
import open3d as o3d


class Object:
    def __init__(self, rgb: np.ndarray, mask: np.ndarray, semantic_ft: np.ndarray, score: float, pcd_points: np.ndarray,
                 pcd_rgb: np.ndarray, camera_pose: np.ndarray):
        """Initialize with first view."""
        self.views = ViewHeap()
        self.views.push(View(rgb, mask, semantic_ft, score, camera_pose))
        self.n_detections = 1
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        self.pcd.colors = o3d.utility.Vector3dVector(pcd_rgb / 255.)
        self.pcd.transform(camera_pose)
        self.centroid = np.mean(pcd_points, axis=0)
        self.semantic_ft = semantic_ft

    def update_semantic_ft(self):
        """Pick the representative semantic vector from the views"""
        ft = [v.semantic_ft for v in self.views]
        ft = np.stack(ft, axis=0)

        mean = np.mean(ft, axis=0)

        self.semantic_ft =  mean/np.linalg.norm(mean, 2)

    def __iadd__(self, other):
        self.views.extend(other.views)
        self.n_detections += other.n_detections
        self.pcd += other.pcd
        if len(self.pcd.points) > 1000:
            self.pcd = self.pcd.uniform_down_sample(every_k_points=2)
        self.centroid = .9 * self.centroid + .1 * other.centroid
        self.update_semantic_ft()

        return self
