from typing import Tuple, List
import numpy as np


def depth_to_point_map(depth: np.ndarray, intrinsics: np.ndarray, camera_pose: np.ndarray):
    H, W = depth.shape
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    y, x = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing="ij")

    x = (x - cx) * depth / fx
    y = (y - cy) * depth / fy
    points = np.stack([x, y, depth], axis=-1)

    return points @ camera_pose[:3, :3].T + camera_pose[:3, 3]


def rgbd_to_object_pcd(
    rgb: np.ndarray,
    depth: np.ndarray,
    masks: List[np.ndarray],
    intrinsics: np.ndarray,
    depth_trunc: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    N, H, W = masks.shape
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    y, x = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing="ij")

    x = (x - cx) * depth / fx
    y = (y - cy) * depth / fy
    points = np.stack([x, y, depth], axis=-1)

    object_pcd_points = []
    object_pcd_rgb = []
    rgb = rgb.reshape((H * W, 3))
    points = points.reshape((H * W, 3))
    masks = masks.reshape((N, H * W))
    not_truncated = ((0 < depth) & (depth < depth_trunc)).reshape((H * W))

    for m in masks:
        keep = m & not_truncated
        object_pcd_points.append(points[keep])
        object_pcd_rgb.append(rgb[keep])

    return object_pcd_points, object_pcd_rgb, points.reshape((H, W, 3))
