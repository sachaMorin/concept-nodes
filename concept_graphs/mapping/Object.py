import numpy as np
import open3d as o3d
from .Segment import Segment
from .SegmentHeap import SegmentHeap
from .pcd_callbacks.PointCloudCallback import PointCloudCallback


class Object:
    def __init__(
        self,
        segment: Segment,
        pcd_points: np.ndarray,
        pcd_rgb: np.ndarray,
        segment_heap_size: int,
        geometry_mode: str,
    ):
        """Initialize with first segment."""
        self.geometry_mode = geometry_mode
        self.segments = SegmentHeap(max_size=segment_heap_size)
        self.segments.push(segment)
        self.n_segments = 1
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        self.pcd.colors = o3d.utility.Vector3dVector(pcd_rgb / 255.0)
        self.pcd.transform(segment.camera_pose)

        self.geometry = None
        self.update_geometry()
        self.semantic_ft = segment.semantic_ft

        self.is_denoised = False
        self.is_downsampled = False

    @property
    def centroid(self):
        return np.mean(self.pcd.points, axis=0)

    def update_geometry(self):
        if self.geometry_mode == "centroid":
            self.geometry = self.centroid
        else:
            raise ValueError(f"Invalid geometry mode {self.geometry_mode}.")

    def update_semantic_ft(self):
        """Pick the representative semantic vector from the segments."""
        ft = [v.semantic_ft for v in self.segments]
        ft = np.stack(ft, axis=0)

        mean = np.mean(ft, axis=0)

        self.semantic_ft = mean / np.linalg.norm(mean, 2)

    def apply_pcd_callback(self, callback: PointCloudCallback):
        self.pcd = callback(self.pcd)
        self.update_geometry()

    def __iadd__(self, other):
        self.segments.extend(other.segments)
        self.n_segments += other.n_segments
        self.pcd += other.pcd
        self.update_geometry()
        self.update_semantic_ft()

        self.is_denoised = False
        self.is_downsampled = False

        return self


class ObjectFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs) -> Object:
        return Object(**kwargs, **self.kwargs)
