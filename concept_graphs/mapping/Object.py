import numpy as np
import open3d as o3d
from .Segment import Segment
from .SegmentHeap import SegmentHeap
from .pcd_callbacks.PointCloudCallback import PointCloudCallback


class Object:
    def __init__(self, segment: Segment, pcd_points: np.ndarray, pcd_rgb: np.ndarray):
        """Initialize with first segment."""
        self.segments = SegmentHeap()
        self.segments.push(segment)
        self.n_segments = 1
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        self.pcd.colors = o3d.utility.Vector3dVector(pcd_rgb / 255.)
        self.pcd.transform(segment.camera_pose)

        self.centroid = None
        self.update_centroid()
        self.semantic_ft = segment.semantic_ft

        self.is_denoised = False
        self.is_downsampled = False

    def update_centroid(self):
        self.centroid = np.mean(self.pcd.points, axis=0)

    def update_semantic_ft(self):
        """Pick the representative semantic vector from the segments."""
        ft = [v.semantic_ft for v in self.segments]
        ft = np.stack(ft, axis=0)

        mean = np.mean(ft, axis=0)

        self.semantic_ft =  mean/np.linalg.norm(mean, 2)

    def apply_pcd_callback(self, callback: PointCloudCallback):
        self.pcd = callback(self.pcd)
        self.update_centroid()

    def __iadd__(self, other):
        self.segments.extend(other.segments)
        self.n_segments += other.n_segments
        self.pcd += other.pcd
        self.update_centroid()
        self.update_semantic_ft()

        self.is_denoised = False
        self.is_downsampled = False

        return self
