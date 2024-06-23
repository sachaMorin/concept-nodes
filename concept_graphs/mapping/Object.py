from typing import Union
import numpy as np
import open3d as o3d
from .Segment import Segment
from .SegmentHeap import SegmentHeap
from .pcd_callbacks.PointCloudCallback import PointCloudCallback


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
        geometry_mode: str,
        semantic_mode: str,
        timestep_created: int,
        n_sample_pcd: int = 20,
        denoising_callback: Union[PointCloudCallback, None] = None,
        downsampling_callback: Union[PointCloudCallback, None] = None,
    ):
        self.segment_heap_size = segment_heap_size
        self.semantic_mode = semantic_mode
        self.geometry_mode = geometry_mode
        self.timestep_created = timestep_created
        self.n_sample_pcd = n_sample_pcd
        self.denoising_callback = denoising_callback
        self.downsampling_callback = downsampling_callback

        self.geometry = None
        self.semantic_ft = None
        self.segments = SegmentHeap(max_size=segment_heap_size)
        self.n_segments = 1
        self.caption = ""

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
                idx = np.random.choice(
                    n_points, self.n_sample_pcd - n_points, replace=True
                )
                self.geometry = np.concatenate([points, points[idx]], axis=0)
            else:
                self.geometry = points[
                    np.random.choice(n_points, self.n_sample_pcd, replace=False)
                ]
        else:
            raise ValueError(f"Invalid geometry mode {self.geometry_mode}.")

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
            self.update_pcd()
            # self.downsample()
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

        rgb_crops = [v.rgb for v in self.segments]
        plot_grid_images(rgb_crops, None, grid_width=3, title=self.caption)


class ObjectFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs) -> Object:
        return Object(**kwargs, **self.kwargs)
