from typing import Union
import numpy as np
import open3d as o3d
from .Segment import Segment
from .SegmentHeap import SegmentHeap
from .pcd_callbacks.PointCloudCallback import PointCloudCallback
import uuid


class Object:
    def __init__(
        self,
        score: float,
        rgb: np.ndarray,
        mask: np.ndarray,
        semantic_ft: np.ndarray,
        camera_pose: np.ndarray,
        pcd_points: np.ndarray,
        pcd_rgb: np.ndarray,
        segment_heap_size: int,
        semantic_mode: str,
        timestep_created: int,
        max_points_pcd: int = 1200,
        denoising_callback: Union[PointCloudCallback, None] = None,
        downsampling_callback: Union[PointCloudCallback, None] = None,
    ):
        self.segment_heap_size = segment_heap_size
        self.semantic_mode = semantic_mode
        self.timestep_created = timestep_created
        self.max_points_pcd = max_points_pcd
        self.denoising_callback = denoising_callback
        self.downsampling_callback = downsampling_callback

        self.pcd = None
        self.pcd_np = None
        self.centroid = None
        self.semantic_ft = None
        self.segments = SegmentHeap(max_size=segment_heap_size)
        self.n_segments = 1
        self.caption = "empty"
        self.tag = "empty"

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

        self.update_geometry_np()
        self.update_semantic_ft()
        self.is_collated = True

        self.id = uuid.uuid4()

    def __repr__(self):
        return f"Object with {len(self.segments)} segments. Detected a total of {self.n_segments} times."

    def update_geometry(self):
        """Pull segment point clouds into one object-level point cloud"""
        points = np.concatenate([s.pcd_points for s in self.segments], axis=0)
        colors = np.concatenate([s.pcd_rgb for s in self.segments], axis=0)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.downsample()
        self.update_geometry_np()

    def update_geometry_np(self):
        self.pcd_np = np.asarray(self.pcd.points)  # No copy
        self.centroid = np.mean(self.pcd_np, axis=0)

        if len(self.pcd_np) > self.max_points_pcd:
            sub = np.random.choice(
                len(self.pcd_np), size=self.max_points_pcd, replace=False
            )
            self.pcd_np = self.pcd_np[sub]

    def update_semantic_ft(self):
        """Pick the representative semantic vector from the segments."""
        ft = [v.semantic_ft for v in self.segments]
        ft = np.stack(ft, axis=0)

        if self.semantic_mode == "mean":
            mean = np.mean(ft, axis=0)
            self.semantic_ft = mean / np.linalg.norm(mean, 2)
        elif self.semantic_mode == "multi":
            if len(ft) < self.segment_heap_size:
                multiply = self.segment_heap_size // len(ft) + 1
                self.semantic_ft = np.concatenate([ft] * multiply, axis=0)
                self.semantic_ft = self.semantic_ft[: self.segment_heap_size]
            else:
                self.semantic_ft = ft

    def collate(self):
        if not self.is_collated:
            self.update_geometry()
            self.update_semantic_ft()

            self.is_collated = True

    def denoise(self):
        if self.denoising_callback is not None:
            self.pcd = self.denoising_callback(self.pcd)

    def downsample(self):
        if self.downsampling_callback is not None:
            self.pcd = self.downsampling_callback(self.pcd)

    def cluster_top_k(self, k: int):
        ft = [v.semantic_ft for v in self.segments]
        ft = np.stack(ft, axis=0)
        mean = np.mean(ft, axis=0, keepdims=True)
        mean /= np.linalg.norm(mean, axis=1, keepdims=True)
        sim = ft @ mean.T
        idx = sim[:, 0].argsort()[-k:]

        new_heap = SegmentHeap(max_size=self.segments.max_size)
        for i in idx:
            new_heap.push(self.segments[i])
        self.segments = new_heap
        self.is_collated = False

    def __iadd__(self, other):
        if self.id == other.id:
            raise Exception("Trying to merge object with self.")

        segment_added = self.segments.extend(other.segments)
        self.n_segments += other.n_segments
        self.timestep_created = min(self.timestep_created, other.timestep_created)

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

    def view_images_caption(self):
        from ..viz.segmentation import plot_grid_images

        rgb_crops = [v.rgb / 255.0 for v in self.segments]
        plot_grid_images(
            rgb_crops, None, grid_width=3, tag=self.tag, caption=self.caption
        )


class RunningAverageObject(Object):
    """CG object from the original paper. Semantic feature average. Append pcd.

    We still use the segment heap to store images and masks, but nothing else."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.semantic_ft = self.segments[0].semantic_ft

        # We track an object-level point cloud so we don't need the segment point clouds
        self.segments[0].pcd_points = None
        self.segments[0].pcd_rgb = None
        self.segments[0].semantic_ft = None

    def update_geometry(self):
        self.downsample()
        self.update_geometry_np()

    def update_semantic_ft(self):
        pass

    def cluster_top_k(self, k: int):
        pass

    def __iadd__(self, other):
        if self.id == other.id:
            raise Exception("Trying to merge object with self.")

        self.segments.extend(other.segments)
        self_ratio = self.n_segments / (self.n_segments + other.n_segments)
        other_ratio = other.n_segments / (self.n_segments + other.n_segments)
        self.semantic_ft = (
            self_ratio * self.semantic_ft + other_ratio * other.semantic_ft
        )
        self.semantic_ft = self.semantic_ft / np.linalg.norm(self.semantic_ft, 2)

        self.pcd += other.pcd

        self.n_segments += other.n_segments
        self.timestep_created = min(self.timestep_created, other.timestep_created)
        self.is_collated = False

        return self


class ObjectFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs) -> Object:
        return Object(**kwargs, **self.kwargs)


class RunningAverageObjectFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs) -> Object:
        return RunningAverageObject(**kwargs, **self.kwargs)
