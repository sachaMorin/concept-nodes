import hydra
from omegaconf import DictConfig
import logging

from concept_graphs.utils import load_map, set_seed
from concept_graphs.viz.utils import similarities_to_rgb
import numpy as np
import open3d as o3d
import copy

# A logger for this file
log = logging.getLogger(__name__)


class CallbackManager:
    def __init__(self, map, ft_extractor):
        self.map = map
        self.ft_extractor = ft_extractor

        # Geometries
        self.pcd = map.pcd_o3d
        self.bbox = map.oriented_bbox_o3d
        self.centroids = map.centroids_o3d

        # Colorings
        self.og_colors = [o3d.utility.Vector3dVector(copy.deepcopy(p.colors)) for p in self.pcd]
        self.sim_query = .5 * np.ones(len(self.pcd))
        self.random_colors = np.random.rand(len(self.pcd), 3)

        # Color centroids
        for c, color in zip(self.centroids, self.random_colors):
            c.paint_uniform_color(color)

        # Similarities
        self.map.semantic_tensor = map.semantic_tensor.to(ft_extractor.device)
        self.self_semantic_sim = self.map.similarity.semantic_similarity(map.semantic_tensor, map.semantic_tensor)
        self.self_geometric_sim = self.map.similarity.geometric_similarity(map.geometry_tensor, map.geometry_tensor)

        # Toggles
        self.bbox_toggle = False
        self.centroid_toggle = False

    def add_geometries(self, vis):
        for geometry in self.pcd:
            vis.add_geometry(geometry)

    def toggle_bbox(self, vis):
        if not self.bbox_toggle:
            for geometry in self.bbox:
                vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in self.bbox:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        self.bbox_toggle = not self.bbox_toggle

    def toggle_centroids(self, vis):
        if not self.centroid_toggle:
            for geometry in self.centroids:
                vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in self.centroids:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        self.centroid_toggle = not self.centroid_toggle

    def toggle_sim(self, vis):
        rgb = similarities_to_rgb(self.sim_query, cmap_name="viridis")
        for p, c, color in zip(self.pcd, self.centroids, rgb):
            p.paint_uniform_color(np.array(color) / 255)
            c.paint_uniform_color(np.array(color) / 255)
            vis.update_geometry(p)
            if self.centroid_toggle:
                vis.update_geometry(c)
        vis.poll_events()
        vis.update_renderer()

    def toggle_random_color(self, vis):
        for p, c, color in zip(self.pcd, self.centroids, self.random_colors):
            p.paint_uniform_color(color)
            c.paint_uniform_color(color)
            vis.update_geometry(p)
            if self.centroid_toggle:
                vis.update_geometry(c)
        vis.poll_events()
        vis.update_renderer()

    def toggle_rgb(self, vis):
        for p, c in zip(self.pcd, self.og_colors):
            p.colors = c
            vis.update_geometry(p)
        vis.poll_events()
        vis.update_renderer()

    def query(self, vis):
        query = input("Enter query: ")
        query_ft = self.ft_extractor.encode_text([query])
        self.sim_query = self.map.similarity.semantic_similarity(self.map.semantic_tensor.float(), query_ft)
        self.sim_query = self.sim_query.squeeze().cpu().numpy()
        # # Calling toggle_sim(vis) is buggy so we just update colors here directly
        rgb = similarities_to_rgb(self.sim_query, cmap_name="viridis")
        for p, c, color in zip(self.pcd, self.centroids, rgb):
            p.paint_uniform_color(np.array(color) / 255)
            c.paint_uniform_color(np.array(color) / 255)
            vis.update_geometry(p)
            if self.centroid_toggle:
                vis.update_geometry(c)
        vis.update_renderer()

    def toggle_inspect(self, vis):
        obj1 = input("First Object Id: ")
        obj2 = input("Second Object Id: ")
        obj1, obj2 = int(obj1), int(obj2)
        for i, (p, c, color) in enumerate(zip(self.pcd, self.centroids, self.random_colors)):
            if i == obj1 or i == obj2:
                p.paint_uniform_color(color)
                c.paint_uniform_color(color)
            else:
                p.paint_uniform_color([0, 0, 0])
                c.paint_uniform_color([0, 0, 0])
            vis.update_geometry(p)
            if self.centroid_toggle:
                vis.update_geometry(c)
        log.info(print(f"Object {obj1}: {self.map[obj1]}"))
        log.info(print(f"Object {obj2}: {self.map[obj2]}"))
        log.info(f"Geometric Similarity: {self.self_geometric_sim[obj1, obj2]}")
        log.info(f"Semantic Similarity: {self.self_semantic_sim[obj1, obj2]}")
        vis.poll_events()
        vis.update_renderer()

    def register_callbacks(self, vis):
        vis.register_key_callback(ord("B"), self.toggle_bbox)
        vis.register_key_callback(ord("C"), self.toggle_centroids)
        vis.register_key_callback(ord("S"), self.toggle_sim)
        vis.register_key_callback(ord("R"), self.toggle_rgb)
        vis.register_key_callback(ord("Z"), self.toggle_random_color)
        vis.register_key_callback(ord("Q"), self.query)
        vis.register_key_callback(ord("I"), self.toggle_inspect)


@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    map = load_map(cfg.map_path)
    log.info(f"Loading map with a total of {len(map)} objects")
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)

    # Callback Manager
    manager = CallbackManager(map, ft_extractor)

    # Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f'Open3D', width=1280, height=720)

    manager.add_geometries(vis)
    manager.register_callbacks(vis)
    vis.run()


if __name__ == "__main__":
    main()
