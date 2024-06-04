from typing import List, Dict, Union
import numpy as np
import torch
import open3d as o3d
from .Segment import Segment
from .Object import Object
from .pcd_callbacks.PointCloudCallback import PointCloudCallback


class ObjectMap:

    def __init__(self, max_centroid_dist: float, semantic_sim_thresh: float,
                 min_segments: int, denoising_callback: Union[PointCloudCallback, None] = None,
                 downsampling_callback: Union[PointCloudCallback, None] = None, device: str = "cpu"):
        self.max_centroid_dist = max_centroid_dist
        self.semantic_sim_thresh = semantic_sim_thresh
        self.min_segments = min_segments
        self.denoising_callback = denoising_callback
        self.downsampling_callback = downsampling_callback
        self.device = device

        self.current_id = 0
        self.objects: Dict[int, Object] = dict()
        self.semantic_ft: torch.Tensor = None
        self.centroids: torch.Tensor = None

        # Map from index in collated arrays(e.g., semantic_ft, centroids) to object key in self.objects
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

    def concat(self, other: 'ObjectMap') -> None:
        """Merge maps without merging objects."""
        for obj in other.objects.values():
            self.append(obj)
        self.collate_geometry()
        self.collate_semantic_ft()

    def collate_geometry(self):
        if len(self):
            centroids = list()
            for obj in self:
                centroids.append(obj.centroid)

            self.centroids = torch.from_numpy(np.stack(centroids, axis=0)).to(self.device)
        else:
            self.centroids = None

    def collate_semantic_ft(self):
        if len(self):
            ft = list()
            for obj in self.objects.values():
                ft.append(obj.semantic_ft)
            self.semantic_ft = torch.from_numpy(np.stack(ft, axis=0)).to(self.device)
        else:
            self.semantic_ft = None

    def collate_keys(self):
        self.key_map = list(self.objects.keys())

    def collate(self):
        self.collate_geometry()
        self.collate_semantic_ft()
        self.collate_keys()

    def from_perception(self, rgb_crops: List[np.ndarray], mask_crops: List[np.ndarray], features: np.ndarray,
                        scores: np.ndarray, pcd_points: List[np.ndarray], pcd_rgb: List[np.ndarray],
                        camera_pose: np.ndarray, is_bg: np.ndarray):
        n_objects = len(rgb_crops)
        assert n_objects == len(mask_crops) == len(features) == len(scores) == len(pcd_points) == len(pcd_rgb) == len(
            is_bg)

        for i in range(len(rgb_crops)):
            if not is_bg[i]:
                segment = Segment(rgb=rgb_crops[i], mask=mask_crops[i], semantic_ft=features[i], score=float(scores[i]),
                                  camera_pose=camera_pose)
                object = Object(segment=segment, pcd_points=pcd_points[i], pcd_rgb=pcd_rgb[i])
                self.append(object)

        self.semantic_ft = torch.from_numpy(features[~is_bg]).to(self.device)
        self.collate_geometry()
        self.collate_keys()

    def similarity(self, other: 'ObjectMap') -> torch.Tensor:
        """Compute similarities with objects from another map."""
        semantic_sim = self.semantic_ft @ other.semantic_ft.t()

        geometric_sim = torch.cdist(self.centroids, other.centroids)
        is_close = geometric_sim < self.max_centroid_dist

        sim = torch.where(is_close, semantic_sim, -1 * torch.ones_like(semantic_sim))

        return sim

    def __iadd__(self, other: 'ObjectMap'):
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self

        sim = self.similarity(other).t()
        merge = (sim > self.semantic_sim_thresh).any(dim=1)
        match = sim.argmax(dim=1).cpu().tolist()

        for i, obj in enumerate(other):
            if merge[i]:
                self[match[i]] += obj
            else:
                self.append(obj)

        self.collate()

        return self

    def self_merge(self):
        if len(self) == 0:
            return self

        sim = self.similarity(self).t()
        sim.fill_diagonal_(-1)  # Avoid self merge attempts
        merge = (sim > self.semantic_sim_thresh).any(dim=1)
        match = sim.argmax(dim=1).cpu().tolist()
        to_delete = list()

        for i, obj in enumerate(self):
            if i not in to_delete and merge[i]:
                j = match[i]
                self[i] += self[j]
                self[j] = self[i]  # Reference to i
                if j not in to_delete:
                    to_delete.append(j)

        for i in to_delete:
            self.pop(i)

        self.collate()

    def filter_min_segments(self):
        self.objects = {k: v for k, v in self.objects.items() if v.n_detections >= self.min_segments}
        self.collate()

    def denoise_pcd(self):
        if self.denoising_callback is not None:
            for obj in self:
                obj.apply_pcd_callback(self.denoising_callback)

    def downsample_pcd(self):
        if self.downsampling_callback is not None:
            for obj in self:
                obj.apply_pcd_callback(self.downsampling_callback)

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
    def centroids_o3d(self) -> List[o3d.geometry.OrientedBoundingBox]:
        if len(self):
            centroids = []
            centroids_np = self.centroids.cpu().numpy()
            for c in centroids_np:
                centroid = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
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
