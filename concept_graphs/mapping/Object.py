from typing import Union
import numpy as np
import open3d as o3d
from .Segment import Segment
from .SegmentHeap import SegmentHeap
from .pcd_callbacks.PointCloudCallback import PointCloudCallback


class Object:
    def __init__(
            self,
            rgb: np.ndarray,
            mask: np.ndarray,
            semantic_ft: np.ndarray,
            camera_pose: np.ndarray,
            score: float,
            pcd_points: np.ndarray,
            pcd_rgb: np.ndarray,
            segment_heap_size: int,
            geometry_mode: str,
            n_sample_pcd: int = 20,
            denoising_callback: Union[PointCloudCallback, None] = None,
            downsampling_callback: Union[PointCloudCallback, None] = None,
    ):
        self.geometry_mode = geometry_mode
        self.n_sample_pcd = n_sample_pcd
        self.denoising_callback = denoising_callback
        self.downsampling_callback = downsampling_callback

        self.geometry = None
        self.semantic_ft = None
        self.segments = SegmentHeap(max_size=segment_heap_size)
        self.n_segments = 1

        # Set our first object-level point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        self.pcd.colors = o3d.utility.Vector3dVector(pcd_rgb / 255.0)
        self.pcd.transform(camera_pose)
        self.downsample()
        self.denoise()

        # Use processed pcd for segment
        pcd_points = np.array(self.pcd.points)
        pcd_rgb = np.array(self.pcd.colors)

        # Create first segment and push to heap
        segment = Segment(
            rgb=rgb,
            mask=mask,
            semantic_ft=semantic_ft,
            camera_pose=camera_pose,
            score=score,
            pcd_points=pcd_points,
            pcd_rgb=pcd_rgb,
        )

        self.segments.push(segment)

        self.update_geometry()
        self.update_semantic_ft()
        self.is_collated = True

    def __repr__(self):
        return f"Object with {len(self.segments)} segments. Detected a total of {self.n_segments} times."

    @property
    def centroid(self):
        return np.mean(self.pcd.points, axis=0)

    def update_pcd(self):
        points = np.concatenate([s.pcd_points for s in self.segments], axis=0)
        colors = np.concatenate([s.pcd_rgb for s in self.segments], axis=0)
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def update_geometry(self):
        # Update geometry
        if self.geometry_mode == "centroid":
            self.geometry = self.centroid
        elif self.geometry_mode == "pcd":
            points = np.asarray(self.pcd.points)
            n_points = points.shape[0]

            if n_points < self.n_sample_pcd:
                # Sample n_sample_pcd - n_points points with replacement
                idx = np.random.choice(n_points, self.n_sample_pcd - n_points, replace=True)
                self.geometry = np.concatenate([points, points[idx]], axis=0)
            else:
                self.geometry = points[np.random.choice(n_points, self.n_sample_pcd, replace=False)]
        else:
            raise ValueError(f"Invalid geometry mode {self.geometry_mode}.")

    def update_semantic_ft(self):
        """Pick the representative semantic vector from the segments."""
        ft = [v.semantic_ft for v in self.segments]
        ft = np.stack(ft, axis=0)

        mean = np.mean(ft, axis=0)

        self.semantic_ft = mean / np.linalg.norm(mean, 2)

    def collate(self):
        if not self.is_collated:
            self.update_pcd()
            self.update_geometry()
            self.update_semantic_ft()

            self.is_collated = True

    def denoise(self):
        if self.denoising_callback is not None:
            self.pcd = self.denoising_callback(self.pcd)

    def downsample(self):
        if self.downsampling_callback is not None:
            self.pcd = self.downsampling_callback(self.pcd)

    def __iadd__(self, other):
        segment_added = self.segments.extend(other.segments)
        self.n_segments += other.n_segments

        if segment_added:
            self.is_collated = False

        return self

    def pcd_to_np(self):
        # Make object pickable
        pcd_points = np.array(self.pcd.points)
        pcd_colors = np.array(self.pcd.colors)
        self.pcd = {"points": pcd_points, "colors": pcd_colors}

    def pcd_to_o3d(self):
        pcd_dict = self.pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_dict["points"])
        pcd.colors = o3d.utility.Vector3dVector(pcd_dict["colors"])
        self.pcd = pcd


class ObjectFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs) -> Object:
        return Object(**kwargs, **self.kwargs)
