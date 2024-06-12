import hydra
from omegaconf import DictConfig
import logging

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.visualization.gui as gui
import copy

from concept_graphs.utils import load_map, set_seed
from concept_graphs.viz.utils import similarities_to_rgb
from concept_graphs.viz.segmentation import plot_grid_images

# A logger for this file
log = logging.getLogger(__name__)


class CallbackManager:
    def __init__(self, map, ft_extractor, mode):
        self.map = map
        self.ft_extractor = ft_extractor
        self.mode = mode

        # Geometries
        self.pcd = map.pcd_o3d
        self.bbox = map.oriented_bbox_o3d
        self.centroids = map.centroids_o3d
        self.pcd_names = [f"pcd_{i}" for i in range(len(map))]
        self.bbox_names = [f"bbox_{i}" for i in range(len(map))]
        self.centroid_names = [f"centroid_{i}" for i in range(len(map))]
        self.label_names = [str(i) for i in range(len(map))]
        self.label_coord = [o.centroid for o in map]

        # Colorings
        self.og_colors = [
            o3d.utility.Vector3dVector(copy.deepcopy(p.colors)) for p in self.pcd
        ]
        self.sim_query = 0.5 * np.ones(len(self.pcd))
        self.random_colors = np.random.rand(len(self.pcd), 3)

        # Color centroids
        for c, color in zip(self.centroids, self.random_colors):
            c.paint_uniform_color(color)

        # Similarities
        self.map.semantic_tensor = map.semantic_tensor.to(ft_extractor.device)
        self.self_semantic_sim = self.map.similarity.semantic_similarity(
            map.semantic_tensor, map.semantic_tensor
        )
        self.map.geometry_tensor = map.geometry_tensor.to(ft_extractor.device)
        self.self_geometric_sim = self.map.similarity.geometric_similarity(
            map.geometry_tensor, map.geometry_tensor
        )

        # Toggles
        self.bbox_toggle = False
        self.centroid_toggle = False
        self.number_toggle = False

    def add_geometries(self, vis, geometry_names, geometries):
        if self.mode == "keycallback":
            for geometry in geometries:
                vis.add_geometry(geometry)
        elif self.mode == "gui":
            for name, geometry in zip(geometry_names, geometries):
                vis.add_geometry(name, geometry)

    def remove_geometries(self, vis, geometry_names, geometries):
        if self.mode == "keycallback":
            for geometry in geometries:
                vis.remove_geometry(geometry)
        elif self.mode == "gui":
            for name in geometry_names:
                vis.remove_geometry(name)

    def update_geometries(self, vis, geometry_names, geometries):
        if self.mode == "keycallback":
            for geometry in geometries:
                vis.update_geometry(geometry)
        elif self.mode == "gui":
            self.remove_geometries(vis, geometry_names, geometries)
            self.add_geometries(vis, geometry_names, geometries)

    def toggle_bbox(self, vis):
        if not self.bbox_toggle:
            self.add_geometries(vis, self.bbox_names, self.bbox)
        else:
            self.remove_geometries(vis, self.bbox_names, self.bbox)
        self.bbox_toggle = not self.bbox_toggle

    def toggle_centroids(self, vis):
        if not self.centroid_toggle:
            self.add_geometries(vis, self.centroid_names, self.centroids)
        else:
            self.remove_geometries(vis, self.centroid_names, self.centroids)
        self.centroid_toggle = not self.centroid_toggle

    def toggle_numbers(self, vis):
        if not self.number_toggle:
            for c, n in zip(self.label_coord, self.label_names):
                vis.add_3d_label(c, n)
        else:
            vis.clear_3d_labels()
        self.number_toggle = not self.number_toggle

    def toggle_sim(self, vis):
        rgb = similarities_to_rgb(self.sim_query, cmap_name="viridis")
        for p, c, color in zip(self.pcd, self.centroids, rgb):
            p.paint_uniform_color(np.array(color) / 255)
            c.paint_uniform_color(np.array(color) / 255)
        self.update_geometries(vis, self.pcd_names, self.pcd)
        if self.centroid_toggle:
            self.update_geometries(vis, self.centroid_names, self.centroids)

    def toggle_random_color(self, vis):
        for p, c, color in zip(self.pcd, self.centroids, self.random_colors):
            p.paint_uniform_color(color)
            c.paint_uniform_color(color)
        self.update_geometries(vis, self.pcd_names, self.pcd)
        if self.centroid_toggle:
            self.update_geometries(vis, self.centroid_names, self.centroids)

    def toggle_rgb(self, vis):
        for p, c in zip(self.pcd, self.og_colors):
            p.colors = c
        self.update_geometries(vis, self.pcd_names, self.pcd)

    def query(self, vis):
        query = input("Enter query: ")
        query_ft = self.ft_extractor.encode_text([query])
        self.sim_query = self.map.similarity.semantic_similarity(
            self.map.semantic_tensor.float(), query_ft
        )
        self.sim_query = self.sim_query.squeeze().cpu().numpy()
        self.toggle_sim(vis)

    def toggle_pairwise_inspect(self, vis):
        obj1 = input("First Object Id: ")
        obj2 = input("Second Object Id: ")
        obj1, obj2 = int(obj1), int(obj2)
        for i, (p, c, color) in enumerate(
            zip(self.pcd, self.centroids, self.random_colors)
        ):
            if i == obj1 or i == obj2:
                p.paint_uniform_color(color)
                c.paint_uniform_color(color)
            else:
                p.paint_uniform_color([0, 0, 0])
                c.paint_uniform_color([0, 0, 0])
        self.update_geometries(vis, self.pcd_names, self.pcd)
        if self.centroid_toggle:
            self.update_geometries(vis, self.centroid_names, self.centroids)
        log.info(f"Object {obj1}: {self.map[obj1]}")
        log.info(f"Object {obj2}: {self.map[obj2]}")
        log.info(f"Geometric Similarity: {self.self_geometric_sim[obj1, obj2]}")
        log.info(f"Semantic Similarity: {self.self_semantic_sim[obj1, obj2]}")

    def view(self, vis):
        obj_id = input("Object Id: ")
        obj_id = int(obj_id)
        obj = self.map[obj_id]
        obj.view_images_caption()
        plt.show()

    def register_callbacks(self, vis):
        if self.mode == "keycallback":
            vis.register_key_callback(ord("R"), self.toggle_rgb)
            vis.register_key_callback(ord("Z"), self.toggle_random_color)
            vis.register_key_callback(ord("S"), self.toggle_sim)
            vis.register_key_callback(ord("B"), self.toggle_bbox)
            vis.register_key_callback(ord("C"), self.toggle_centroids)
            vis.register_key_callback(ord("Q"), self.query)
            vis.register_key_callback(ord("P"), self.toggle_pairwise_inspect)
            vis.register_key_callback(ord("V"), self.view)
        else:
            vis.add_action("RGB", self.toggle_rgb)
            vis.add_action("Random Color", self.toggle_random_color)
            vis.add_action("Similarity", self.toggle_sim)
            vis.add_action("Toggle Bbox", self.toggle_bbox)
            vis.add_action("Toggle Centroid", self.toggle_centroids)
            vis.add_action("Toggle Number", self.toggle_numbers)
            vis.add_action("CLIP Query", self.query)
            vis.add_action("Pairwise Inspect", self.toggle_pairwise_inspect)
            vis.add_action("View Segments", self.view)


@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    map = load_map(cfg.map_path)
    log.info(f"Loading map with a total of {len(map)} objects")
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)

    # Callback Manager
    manager = CallbackManager(map, ft_extractor, mode=cfg.mode)

    # Visualizer
    if cfg.mode == "keycallback":
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"Open3D", width=1280, height=720)

        manager.add_geometries(vis, manager.pcd_names, manager.pcd)
        manager.register_callbacks(vis)
        vis.run()
    elif cfg.mode == "gui":
        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        vis.set_background([1.0, 1.0, 1.0, 1.0], bg_image=None)
        vis.show_settings = True
        vis.show_skybox(False)
        vis.enable_raw_mode(True)
        manager.add_geometries(vis, manager.pcd_names, manager.pcd)
        manager.register_callbacks(vis)
        # for idx in range(0, len(points.points)):
        #     vis.add_3d_label(points.points[idx], "{}".format(idx))
        vis.reset_camera_to_default()

        app.add_window(vis)
        app.run()

    else:
        raise ValueError("Invalid mode.")


if __name__ == "__main__":
    main()
