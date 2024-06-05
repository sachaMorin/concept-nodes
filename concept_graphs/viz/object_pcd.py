from typing import List, Union, Tuple
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
from .utils import similarities_to_rgb


def visualize_object_pcd(
    pcd_points: List[np.ndarray],
    pcd_rgb: List[Union[np.ndarray, Tuple[int, int, int]]] = None,
):
    geometries = []
    for i, pcd in enumerate(pcd_points):
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
        if pcd_rgb is not None:
            color = pcd_rgb[i]
            if type(color) is np.ndarray:
                pcd_o3d.colors = o3d.utility.Vector3dVector(pcd_rgb[i] / 255)
            elif type(color) is tuple:
                color = np.array(color) / 255
                pcd_o3d.paint_uniform_color(color)
            else:
                raise ValueError()
        else:
            # Sample random color for whole object
            pcd_o3d.paint_uniform_color(
                [np.random.rand(), np.random.rand(), np.random.rand()]
            )

        geometries.append(pcd_o3d)

    o3d.visualization.draw_geometries(geometries)


def visualize_object_pcd_similarities(
    pcd_points: List[np.ndarray], similarities: np.ndarray
):
    rgb_sim = similarities_to_rgb(similarities, "viridis")
    visualize_object_pcd(pcd_points, rgb_sim)
