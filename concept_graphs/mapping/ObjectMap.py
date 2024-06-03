from typing import List, Dict
import numpy as np
import torch
from .Object import Object
import open3d as o3d


class ObjectMap:

    def __init__(self, max_centroid_dist: float = 0.5, semantic_sim_thresh: float = 0.5, device: str = "cpu"):
        self.current_id = 0
        self.objects: Dict[int, Object] = dict()
        self.semantic_ft: torch.Tensor = None
        self.centroids: torch.Tensor = None
        self.max_centroid_dist = max_centroid_dist
        self.semantic_sim_thresh = semantic_sim_thresh
        self.device = device

    def __len__(self):
        return len(self.objects)

    @property
    def pcd(self) -> List[o3d.geometry.PointCloud]:
        return [o.pcd for o in self.objects.values()]

    def append(self, obj: Object):
        self.objects[self.current_id] = obj
        self.current_id += 1

    def from_perception(rgb_crops: List[np.ndarray], mask_crops: List[np.ndarray], features: np.ndarray,
                        scores: np.ndarray, pcd_points: List[np.ndarray], pcd_rgb: List[np.ndarray],
                        camera_pose: np.ndarray, device:str, **kwargs):
        map = ObjectMap(device=device, **kwargs)
        n_objects = len(rgb_crops)

        assert n_objects == len(mask_crops) == len(features) == len(scores) == len(pcd_points) == len(pcd_rgb)

        for i in range(len(rgb_crops)):
            map.append(Object(rgb_crops[i], mask_crops[i], features[i], float(scores[i]), pcd_points[i], pcd_rgb[i],
                              camera_pose))

        map.semantic_ft = torch.from_numpy(features).to(device)
        map.collate_geometry()

        return map

    def collate_geometry(self):
        centroids = list()
        for obj in self.objects.values():
            centroids.append(obj.centroid)

        self.centroids = torch.from_numpy(np.stack(centroids, axis=0)).to(self.device)

    def collate_semantic_ft(self):
        ft = list()
        for obj in self.objects.values():
            ft.append(obj.semantic_ft)
        self.semantic_ft = torch.from_numpy(np.stack(ft, axis=0)).to(self.device)

    def draw_geometries(self, random_colors: bool = False) -> None:
        pcd = self.pcd
        if random_colors:
            for p in pcd:
                p.paint_uniform_color(np.random.rand(3))

        o3d.visualization.draw_geometries(pcd)

    def concat(self, other: 'ObjectMap') -> None:
        """Merge maps without merging objects."""
        for obj in other.objects.values():
            self.append(obj)
        self.collate_geometry()
        self.collate_semantic_ft()

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

        for i, obj in enumerate(other.objects.values()):
            if merge[i]:
                self.objects[match[i]] += obj
            else:
                self.append(obj)

        self.collate_semantic_ft()
        self.collate_geometry()

        return self

    def save_object_grids(self, save_dir: str):
        import matplotlib.pyplot as plt
        from ..viz.segmentation import plot_grid_images
        for i, obj in enumerate(self.objects.values()):
            rgb_crops = [v.rgb for v in obj.views]
            masks = [v.mask for v in obj.views]
            plot_grid_images(rgb_crops, masks)
            plt.savefig(f"{save_dir}/{i}.png")
            plt.close()
