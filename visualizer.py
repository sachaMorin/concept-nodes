import hydra
from omegaconf import DictConfig
import logging

from concept_graphs.utils import load_map
from concept_graphs.viz.utils import similarities_to_rgb
import numpy as np
import open3d as o3d

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig):
    map = load_map(cfg.map_path)

    pcd = map.pcd_o3d
    bbox = map.oriented_bbox_o3d
    centroids = map.centroids_o3d

    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)
    query  = "somewhere to sit"
    query_ft = ft_extractor.encode_text([query])
    sim = query_ft @ map.semantic_tensor.T
    sim = sim.squeeze().cpu().numpy()
    rgb = similarities_to_rgb(sim, cmap_name="viridis")

    for p, c, color in zip(pcd, centroids, rgb):
        p.paint_uniform_color(np.array(color)/255)
        c.paint_uniform_color(np.array(color)/255)

    o3d.visualization.draw_geometries(pcd + centroids)

if __name__ == "__main__":
    main()

