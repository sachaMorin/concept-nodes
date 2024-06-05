import open3d as o3d
from .PointCloudCallback import PointCloudCallback
import numpy as np


class RemoveStatOutlier(PointCloudCallback):
    def __init__(self, n_neighbors=20, std_ratio=2.0):
        self.n_neighbors = n_neighbors
        self.std_ratio = std_ratio

    def __call__(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        pcd, _ = pcd.remove_statistical_outlier(self.n_neighbors, self.std_ratio)
        return pcd


class RemoveRadiusOutlier(PointCloudCallback):
    def __init__(self, n_points=20, radius=0.02):
        self.n_points = n_points
        self.radius = radius

    def __call__(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if len(pcd.points) < self.n_points:
            return pcd

        pcd, _ = pcd.remove_radius_outlier(nb_points=self.n_points, radius=self.radius)
        return pcd


class DBSCAN(PointCloudCallback):
    def __init__(self, eps=0.02, min_points=10):
        self.eps = eps
        self.min_points = min_points

    def __call__(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if len(pcd.points) < self.min_points:
            return pcd

        labels = pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points)
        labels_unique, counts = np.unique(labels, return_counts=True)
        max_label = labels_unique[np.argmax(counts)]
        mask = labels == max_label

        return pcd.select_by_index(np.where(mask)[0])


class UniformDownSampling(PointCloudCallback):
    def __init__(self, every_k_points=2):
        self.every_k_points = every_k_points

    def __call__(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return pcd.uniform_down_sample(self.every_k_points)


class VoxelDownSampling(PointCloudCallback):
    def __init__(self, voxel_size=0.02):
        self.voxel_size = voxel_size

    def __call__(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return pcd.voxel_down_sample(self.voxel_size)
