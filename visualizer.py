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


@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    map = load_map(cfg.map_path)
    log.info(f"Loading map with a total of {len(map)} objects")

    # Geometries
    pcd = map.pcd_o3d
    bbox = map.oriented_bbox_o3d
    centroids = map.centroids_o3d

    # Colorings
    og_colors = [o3d.utility.Vector3dVector(copy.deepcopy(p.colors)) for p in pcd]
    sim_query = .5 * np.ones(len(pcd))
    random_colors = np.random.rand(len(pcd), 3)

    # Similarities
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)
    map.semantic_tensor = map.semantic_tensor.to(ft_extractor.device)
    similarity_cb = map.similarity
    self_semantic_sim = similarity_cb.semantic_similarity(map.semantic_tensor, map.semantic_tensor)
    self_geometric_sim = similarity_cb.geometric_similarity(map.geometry_tensor, map.geometry_tensor)

    # Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f'Open3D', width=1280, height=720)

    for geometry in pcd + centroids:
        vis.add_geometry(geometry)

    # Bbox
    bbox_toggle = False

    def toggle_bbox(vis):
        nonlocal bbox_toggle
        bbox_toggle = not bbox_toggle
        if bbox_toggle:
            for geometry in bbox:
                vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in bbox:
                vis.remove_geometry(geometry, reset_bounding_box=False)

    # Colorings
    def toggle_sim(vis):
        nonlocal sim_query
        rgb = similarities_to_rgb(sim_query, cmap_name="viridis")
        for p, c, color in zip(pcd, centroids, rgb):
            p.paint_uniform_color(np.array(color)/255)
            c.paint_uniform_color(np.array(color)/255)
            vis.update_geometry(p)
            vis.update_geometry(c)
        vis.poll_events()
        vis.update_renderer()

    def toggle_random_color(vis):
        nonlocal random_colors
        for p, c, color in zip(pcd, centroids, random_colors):
            p.paint_uniform_color(color)
            c.paint_uniform_color(color)
            vis.update_geometry(p)
            vis.update_geometry(c)
        vis.poll_events()
        vis.update_renderer()

    def toggle_rgb(vis):
        nonlocal og_colors
        for p, c in zip(pcd, og_colors):
            p.colors = c
            vis.update_geometry(p)
        vis.poll_events()
        vis.update_renderer()

    # Text query
    def query(vis):
        nonlocal ft_extractor
        nonlocal sim_query
        nonlocal similarity_cb
        query = input("Enter query: ")
        query_ft = ft_extractor.encode_text([query])
        sim_query = similarity_cb.semantic_similarity(map.semantic_tensor.float(), query_ft)
        sim_query = sim_query.squeeze().cpu().numpy()
        # # Calling toggle_sim(vis) is buggy so we just update colors here directly
        rgb = similarities_to_rgb(sim_query, cmap_name="viridis")
        for p, c, color in zip(pcd, centroids, rgb):
            p.paint_uniform_color(np.array(color)/255)
            c.paint_uniform_color(np.array(color)/255)
            vis.update_geometry(p)
            vis.update_geometry(c)
        vis.update_renderer()

    def toggle_inspect(vis):
        nonlocal random_colors
        nonlocal self_geometric_sim
        nonlocal self_semantic_sim
        obj1 = input("First Object Id: ")
        obj2 = input("Second Object Id: ")
        obj1, obj2 = int(obj1), int(obj2)
        for i, (p, c, color) in enumerate(zip(pcd, centroids, random_colors)):
            if i == obj1 or i == obj2:
                p.paint_uniform_color(color)
                c.paint_uniform_color(color)
            else:
                p.paint_uniform_color([0, 0, 0])
                c.paint_uniform_color([0, 0, 0])
            vis.update_geometry(p)
            vis.update_geometry(c)
        log.info(print(f"Object {obj1}: {map[obj1]}"))
        log.info(print(f"Object {obj2}: {map[obj2]}"))
        log.info(f"Geometric Similarity: {self_geometric_sim[obj1, obj2]}")
        log.info(f"Semantic Similarity: {self_semantic_sim[obj1, obj2]}")
        vis.poll_events()
        vis.update_renderer()

    vis.register_key_callback(ord("B"), toggle_bbox)
    vis.register_key_callback(ord("S"), toggle_sim)
    vis.register_key_callback(ord("R"), toggle_rgb)
    vis.register_key_callback(ord("Z"), toggle_random_color)
    vis.register_key_callback(ord("Q"), query)
    vis.register_key_callback(ord("I"), toggle_inspect)
    vis.run()

if __name__ == "__main__":
    main()

