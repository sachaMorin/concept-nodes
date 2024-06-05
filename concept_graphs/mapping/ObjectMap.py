from typing import List, Dict, Union, Tuple
import numpy as np
import torch
import open3d as o3d
from .Segment import Segment
from .Object import Object, ObjectFactory
from .pcd_callbacks.PointCloudCallback import PointCloudCallback
from .similarities.match_similarities import match_similarities


class ObjectMap:

    def __init__(
        self,
        geometric_sim_thresh: float,
        semantic_sim_thresh: float,
        min_segments: int,
        geometry_mode: str,
        object_factory: ObjectFactory,
        denoising_callback: Union[PointCloudCallback, None] = None,
        downsampling_callback: Union[PointCloudCallback, None] = None,
        self_merge_every: int = -1,
        filter_min_every: int = -1,
        denoise_pcd_every: int = -1,
        downsample_pcd_every: int = -1,
        device: str = "cpu",
    ):
        self.geometric_sim_thresh = geometric_sim_thresh
        self.semantic_sim_thresh = semantic_sim_thresh
        self.min_segments = min_segments
        self.geometry_mode = geometry_mode
        self.object_factory = object_factory
        self.denoising_callback = denoising_callback
        self.downsampling_callback = downsampling_callback
        self.self_merge_every = self_merge_every
        self.filter_min_every = filter_min_every
        self.denoise_pcd_every = denoise_pcd_every
        self.downsample_pcd_every = downsample_pcd_every
        self.device = device

        self.current_id = 0
        self.n_updates = 0
        self.objects: Dict[int, Object] = dict()

        self.semantic_tensor: torch.Tensor = None
        self.geometry_tensor: torch.Tensor = None

        # Map from index in collated tensors (e.g., semantic_tensor, geometry_tensor) to object key in self.objects
        self.key_map = []

    def __len__(self):
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects.values())

    def __getitem__(self, item):
        return self.objects[self.key_map[item]]

    def __setitem__(self, key, value):
        self.objects[self.key_map[key]] = value

    def append(self, obj: Object) -> None:
        self.objects[self.current_id] = obj
        self.current_id += 1
        self.key_map.append(self.current_id)

    def pop(self, key) -> Object:
        return self.objects.pop(self.key_map[key])

    def concat(self, other: "ObjectMap") -> None:
        """Merge maps without merging objects."""
        for obj in other.objects.values():
            self.append(obj)
        self.collate_geometry()
        self.collate_semantic_ft()

    def collate_geometry(self):
        if len(self):
            geometries = list()
            for obj in self:
                geometries.append(obj.geometry)

            self.geometry_tensor = torch.from_numpy(np.stack(geometries, axis=0)).to(
                self.device
            )
        else:
            self.geometry_tensor = None

    def collate_semantic_ft(self):
        if len(self):
            ft = list()
            for obj in self.objects.values():
                ft.append(obj.semantic_ft)
            self.semantic_tensor = torch.from_numpy(np.stack(ft, axis=0)).to(
                self.device
            )
        else:
            self.semantic_tensor = None

    def collate_keys(self):
        self.key_map = list(self.objects.keys())

    def collate(self):
        self.collate_geometry()
        self.collate_semantic_ft()
        self.collate_keys()

    def from_perception(
        self,
        rgb_crops: List[np.ndarray],
        mask_crops: List[np.ndarray],
        features: np.ndarray,
        scores: np.ndarray,
        pcd_points: List[np.ndarray],
        pcd_rgb: List[np.ndarray],
        camera_pose: np.ndarray,
        is_bg: np.ndarray,
    ):
        n_objects = len(rgb_crops)
        assert (
            n_objects
            == len(mask_crops)
            == len(features)
            == len(scores)
            == len(pcd_points)
            == len(pcd_rgb)
            == len(is_bg)
        )

        for i in range(len(rgb_crops)):
            if not is_bg[i]:
                segment = Segment(
                    rgb=rgb_crops[i],
                    mask=mask_crops[i],
                    semantic_ft=features[i],
                    score=float(scores[i]),
                    camera_pose=camera_pose,
                )
                object = self.object_factory(
                    segment=segment, pcd_points=pcd_points[i], pcd_rgb=pcd_rgb[i]
                )
                self.append(object)

        self.semantic_tensor = torch.from_numpy(features[~is_bg]).to(self.device)
        self.collate_geometry()
        self.collate_keys()

    def match_similarities(
        self, other: "ObjectMap", mask_diagonal: bool = False
    ) -> Tuple[List[bool], List[int]]:
        """Compute similarities with objects from another map."""
        return match_similarities(
            main_semantic=self.semantic_tensor,
            main_geometry=self.geometry_tensor,
            other_semantic=other.semantic_tensor,
            other_geometry=other.geometry_tensor,
            semantic_sim_thresh=self.semantic_sim_thresh,
            geometric_sim_thresh=self.geometric_sim_thresh,
            geometry_mode=self.geometry_mode,
            mask_diagonal=mask_diagonal,
        )

    def __iadd__(self, other: "ObjectMap"):
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self

        merge, match_idx = self.match_similarities(other)

        for i, obj in enumerate(other):
            if merge[i]:
                self[match_idx[i]] += obj
            else:
                self.append(obj)

        self.collate()

        self.n_updates += 1
        self.check_processing()

        return self

    def self_merge(self):
        if len(self) == 0:
            return self

        merge, match_idx = self.match_similarities(self, mask_diagonal=True)
        to_delete = list()

        for i, obj in enumerate(self):
            if i not in to_delete and merge[i]:
                j = match_idx[i]
                self[i] += self[j]
                self[j] = self[i]  # Reference to i
                if j not in to_delete:
                    to_delete.append(j)

        for i in to_delete:
            self.pop(i)

        self.collate()

    def filter_min_segments(self):
        self.objects = {
            k: v for k, v in self.objects.items() if v.n_segments >= self.min_segments
        }
        self.collate()

    def denoise_pcd(self):
        if self.denoising_callback is not None:
            for obj in self:
                if not obj.is_denoised:
                    obj.apply_pcd_callback(self.denoising_callback)
                    obj.is_denoised = True

    def downsample_pcd(self):
        if self.downsampling_callback is not None:
            for obj in self:
                if not obj.is_downsampled:
                    obj.apply_pcd_callback(self.downsampling_callback)
                    obj.is_downsampled = True

    def check_processing(self):
        if self.filter_min_every > 0 and self.n_updates % self.filter_min_every == 0:
            self.filter_min_segments()
        if (
            self.downsample_pcd_every > 0
            and self.n_updates % self.downsample_pcd_every == 0
        ):
            self.downsample_pcd()
        if self.denoise_pcd_every > 0 and self.n_updates % self.denoise_pcd_every == 0:
            self.denoise_pcd()
        if self.self_merge_every > 0 and self.n_updates % self.self_merge_every == 0:
            self.self_merge()

    def save_object_grids(self, save_dir: str):
        import matplotlib.pyplot as plt
        from ..viz.segmentation import plot_grid_images

        for i, obj in enumerate(self):
            rgb_crops = [v.rgb for v in obj.segments]
            masks = [v.mask for v in obj.segments]
            plot_grid_images(rgb_crops, None, grid_width=3)
            plt.savefig(f"{save_dir}/{i}.png")
            plt.close()

    @property
    def pcd_o3d(self) -> List[o3d.geometry.PointCloud]:
        return [o.pcd for o in self]

    @property
    def oriented_bbox_o3d(self) -> List[o3d.geometry.OrientedBoundingBox]:
        bbox = [o.pcd.get_oriented_bounding_box() for o in self]

        # Change color to black
        for b in bbox:
            b.color = (0, 0, 0)

        return bbox

    @property
    def centroids_o3d(self) -> List[o3d.geometry.TriangleMesh]:
        if len(self):
            centroids = []
            for obj in self:
                centroid = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                c = obj.centroid
                centroid.translate(c)
                centroids.append(centroid)
        else:
            centroids = []

        return centroids

    def draw_geometries(self, random_colors: bool = False) -> None:
        pcd = self.pcd_o3d
        centroids = self.centroids_o3d
        bbox = self.oriented_bbox_o3d

        if random_colors:
            for p, c in zip(pcd, centroids):
                color = np.random.rand(3)
                p.paint_uniform_color(color)
                c.paint_uniform_color(color)

        o3d.visualization.draw_geometries(pcd + centroids + bbox)

    def save(self, path: str) -> None:
        import pickle

        for obj in self:
            obj.pcd_to_np()

        with open(path, "wb") as f:
            pickle.dump(self, f)
