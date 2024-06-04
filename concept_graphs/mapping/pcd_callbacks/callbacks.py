from .PointCloudCallback import PointCloudCallback


class RemoveStatOutlier(PointCloudCallback):
    def __init__(self, n_neighbors=20, std_ratio=2.0):
        self.n_neighbors = n_neighbors
        self.std_ratio = std_ratio

    def __call__(self, pcd):
        pcd, _ = pcd.remove_statistical_outlier(self.n_neighbors, self.std_ratio)
        return pcd


class UniformDownSampling(PointCloudCallback):
    def __init__(self, every_k_points=2):
        self.every_k_points = every_k_points

    def __call__(self, pcd):
        return pcd.uniform_down_sample(self.every_k_points)


class VoxelDownSampling(PointCloudCallback):
    def __init__(self, voxel_size=0.02):
        self.voxel_size = voxel_size

    def __call__(self, pcd):
        return pcd.voxel_down_sample(self.voxel_size)
