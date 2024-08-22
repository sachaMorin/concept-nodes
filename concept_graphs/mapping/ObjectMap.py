from typing import List, Dict, Tuple
import numpy as np
import torch
import open3d as o3d
from .Object import Object, ObjectFactory
from .similarity.Similarity import Similarity
from .utils import pairs_to_connected_components


class ObjectMap:

    def __init__(
            self,
            similarity: Similarity,
            object_factory: ObjectFactory,
            n_min_segments: int,
            min_points_pcd: int,
            grace_min_segments: int,
            filter_min_every: int,
            collate_objects_every: int,
            self_merge_every: int,
            denoise_every: int,
            downsample_every: int,
            device: str = "cpu",
    ):
        self.similarity = similarity
        self.n_min_segments = n_min_segments
        self.min_points_pcd = min_points_pcd
        self.grace_min_segments = grace_min_segments
        self.filter_min_every = filter_min_every
        self.object_factory = object_factory
        self.collate_objects_every = collate_objects_every
        self.self_merge_every = self_merge_every
        self.denoise_every = denoise_every
        self.downsample_every = downsample_every
        self.device = device

        self.current_id = 0
        self.n_updates = 0
        self.objects: Dict[int, Object] = dict()

        self.semantic_tensor: torch.Tensor = None
        self.geometry_tensor: torch.Tensor = None

        # Map from index in collated tensors (e.g., semantic_tensor, geometry_tensor) to object key in self.objects
        self.key_map = []

    def to(self, device: str):
        self.device = device
        if self.semantic_tensor is not None:
            self.semantic_tensor = self.semantic_tensor.to(device)
        if self.geometry_tensor is not None:
            self.geometry_tensor = self.geometry_tensor.to(device)

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
                # Transform pcd_points with camera_pose
                object = self.object_factory(
                    rgb=rgb_crops[i],
                    mask=mask_crops[i],
                    semantic_ft=features[i],
                    pcd_points=pcd_points[i],
                    pcd_rgb=pcd_rgb[i],
                    camera_pose=camera_pose,
                    score=float(scores[i]),
                    timestep_created=self.n_updates,
                )
                self.append(object)

        self.collate()

    def similarity_with(
            self, other: "ObjectMap", mask_diagonal: bool,
    ) -> Tuple[List[bool], List[int]]:
        """Compute similarities with objects from another map."""
        mergeable, merge_idx = self.similarity(
            main_semantic=self.semantic_tensor,
            main_geometry=self.geometry_tensor,
            other_semantic=other.semantic_tensor,
            other_geometry=other.geometry_tensor,
            mask_diagonal=mask_diagonal,
        )

        return mergeable, merge_idx

    def __iadd__(self, other: "ObjectMap"):
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self

        mergeable, merge_idx = self.similarity_with(other, mask_diagonal=False)

        for i, obj in enumerate(other):
            obj.timestep_created = self.n_updates
            if mergeable[i]:
                self[merge_idx[i]] += obj
            else:
                self.append(obj)

        self.collate()

        self.n_updates += 1
        self.check_processing()

        return self

    def self_merge(self):
        if len(self) == 0:
            return self

        mergeable, merge_idx = self.similarity_with(self, mask_diagonal=True)

        if not np.any(mergeable):
            return

        n = len(self)
        pairs = [(i, merge_idx[i]) for i in range(n) if mergeable[i]]
        connected_components = pairs_to_connected_components(pairs, n)

        has_been_merged = list()
        for component in connected_components:
            if len(component) > 1:
                for j in range(1, len(component)):
                    self[component[0]] += self[component[j]]
                    has_been_merged.append(component[j])

        for i in has_been_merged:
            self.pop(i)

        self.collate_objects()
        self.collate()

    def filter_min_segments(self, n_min_segments: int = -1, grace: bool = True):
        if n_min_segments < 0:
            n_min_segments = self.n_min_segments
        new_objects = {}
        for k, obj in self.objects.items():
            if obj.n_segments >= n_min_segments or (
                    grace and (self.n_updates - obj.timestep_created < self.grace_min_segments)):
                new_objects[k] = obj
        self.objects = new_objects
        self.collate()

    def filter_min_points_pcd(self, min_points_pcd: int = -1):
        if min_points_pcd < 0:
            min_points_pcd = self.min_points_pcd
        new_objects = {}
        for k, obj in self.objects.items():
            if len(obj.pcd.points) >= min_points_pcd:
                new_objects[k] = obj
        self.objects = new_objects
        self.collate()

    def collate_objects(self):
        for obj in self:
            obj.collate()

    def denoise_objects(self):
        for obj in self:
            obj.denoise()

    def downsample_objects(self):
        for obj in self:
            obj.downsample()

    def check_processing(self):
        if self.downsample_every > 0 and self.n_updates % self.downsample_every == 0:
            self.downsample_objects()
        if self.denoise_every > 0 and self.n_updates % self.denoise_every == 0:
            self.denoise_objects()
        if self.filter_min_every > 0 and self.n_updates % self.filter_min_every == 0:
            self.filter_min_segments()
            self.filter_min_points_pcd()
        if (
                self.collate_objects_every > 0
                and self.n_updates % self.collate_objects_every == 0
        ):
            self.collate_objects()
        if self.self_merge_every > 0 and self.n_updates % self.self_merge_every == 0:
            self.self_merge()

    def save_object_grids(self, save_dir: str):
        import matplotlib.pyplot as plt

        for i, obj in enumerate(self):
            obj.view_images_caption()
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
