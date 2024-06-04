import open3d as o3d


class PointCloudCallback:
    def __call__(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        raise NotImplementedError
