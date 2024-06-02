from typing import Tuple, List
import numpy as np


def rgbd_to_object_pcd(rgb: np.ndarray, depth: np.ndarray, masks: np.ndarray, intrinsics: np.ndarray) -> Tuple[
    List[np.ndarray], List[np.ndarray]]:
    N, H, W = masks.shape
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    y, x = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing='ij')

    x = (x - cx) * depth / fx
    y = (y - cy) * depth / fy
    points = np.stack([x, y, depth], axis=-1)

    object_pcd_points = []
    object_pcd_rgb = []
    rgb = rgb.reshape((H * W, 3))
    points = points.reshape((H * W, 3))
    masks = masks.reshape((N, H * W))

    for m in masks:
        object_pcd_points.append(points[m])
        object_pcd_rgb.append(rgb[m])


    return object_pcd_points, object_pcd_rgb
