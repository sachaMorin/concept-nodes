import os
from typing import Dict, List, Union
import torch
import numpy as np
from pathlib import Path
import copy
from .ft_extraction.FeatureExtractor import FeatureExtractor
from .segmentation.SegmentationModel import SegmentationModel
from .rgbd_to_pcd import depth_to_point_map
from .segmentation.utils import (
    extract_rgb_crops,
    extract_mask_crops,
    safe_bbox_inflate,
    mask_subtract_contained,
)


def filter_list(list: List, mask: List[bool]):
    return [o for (o, m) in zip(list, mask) if m]


class PerceptionPipeline:
    def __init__(
        self,
        segmentation_model: SegmentationModel,
        segment_scoring_method: str,
        ft_extractor: FeatureExtractor,
        inflate_bbox_px: int,
        depth_trunc: float,
        mask_subtract_contained: bool,
        crop_bg_color: Union[int, None] = None,
        min_mask_area_px: int = 50,
        min_points_pcd: int = 50,
        debug_images: bool = False,
        debug_dir: str = ".",
    ):
        self.segmentation_model = segmentation_model
        self.segment_scoring_method = segment_scoring_method
        self.ft_extractor = ft_extractor
        self.depth_trunc = depth_trunc
        self.inflate_bbox_px = inflate_bbox_px
        self.crop_bg_color = crop_bg_color
        self.mask_subtract_contained = mask_subtract_contained
        self.min_mask_area_px = min_mask_area_px
        self.min_points_pcd = min_points_pcd
        self.debug_images = debug_images
        self.debug_dir = Path(debug_dir)
        self.debug_counter = 0

        if self.debug_images:
            os.makedirs(self.debug_dir / "segments", exist_ok=True)

    def __call__(
        self, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, camera_pose: np.ndarray,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        object_masks, bbox, conf = self.segmentation_model(rgb)

        if object_masks is None or bbox is None or conf is None:
            return None

        # Penalty if original bbox touches image border
        h, w = rgb.shape[:2]
        touches_left = bbox[:, 0] <= 0
        touches_right = bbox[:, 2] >= w - 1
        touches_top = bbox[:, 1] <= 0
        touches_bottom = bbox[:, 3] >= h - 1
        touches_border = touches_left | touches_right | touches_top | touches_bottom
        touches_border = touches_border.cpu().numpy()

        if self.inflate_bbox_px > 0:
            bbox = safe_bbox_inflate(
                bbox, self.inflate_bbox_px, rgb.shape[1], rgb.shape[0]
            )

        if self.mask_subtract_contained:
            object_masks = mask_subtract_contained(bbox, object_masks)

        object_masks, bbox, conf = (
            object_masks.cpu().numpy(),
            bbox.cpu().numpy(),
            conf.cpu().numpy(),
        )

        areas = object_masks.sum(axis=-1).sum(axis=-1)

        # Choose scoring method
        if self.segment_scoring_method == "confidence":
            scores = conf
        elif self.segment_scoring_method == "area":
            scores = areas
        else:
            raise ValueError(
                f"Invalid segment scoring method: {self.segment_scoring_method}"
            )

        # Penalty for touching border
        if touches_border.any():
            scores[touches_border] = scores[touches_border] * 0.10

        # Segment filtering
        keep = areas > self.min_mask_area_px
        object_masks, bbox, scores = object_masks[keep], bbox[keep], scores[keep]

        # Crops and Feature extraction
        object_mask_crops = extract_mask_crops(object_masks, bbox)
        rgb_crops = extract_rgb_crops(rgb, bbox, object_mask_crops)
        rgb_crops_bg = extract_rgb_crops(rgb, bbox, object_mask_crops, self.crop_bg_color)

        point_map = depth_to_point_map(depth, intrinsics, camera_pose)
        point_map_crops = extract_mask_crops([point_map] * len(bbox), bbox) 

        features = self.ft_extractor(rgb_crops_bg)
        features = features.cpu().numpy()
        
        # Create point cloud masks
        # 0: Invalid points.
        # 1: Object points with invalid depth.
        # 2: Object points with valid depth.
        masks = []
        depth_crops = extract_mask_crops([depth] * len(bbox), bbox)
        for m, d in zip(object_mask_crops, depth_crops):
            mask = np.zeros_like(m, dtype=np.uint8)
            mask[m] = 1
            valid_depth = (d > 0) & (d < self.depth_trunc)
            mask[m & valid_depth] = 2
            masks.append(mask)

        if self.debug_images:
            from concept_graphs.viz.segmentation import plot_segments
            import matplotlib.pyplot as plt

            img_name = str(self.debug_counter).zfill(7) + ".png"
            plot_segments(rgb, torch.from_numpy(object_masks))
            plt.savefig(self.debug_dir / "segments" / img_name)
            plt.close()
            self.debug_counter += 1

        # Filter empty point clouds. They can happen at this stage because of depth truncation
        keep = [m.sum() > self.min_points_pcd for m in object_masks]

        result = dict(
            rgb_crops=filter_list(rgb_crops, keep),
            mask_crops=filter_list(masks, keep),
            point_map_crops=filter_list(point_map_crops, keep),
            features=features[keep],
            scores=scores[keep],
            camera_pose=camera_pose,
        )

        # TEMP FIX. This seems to help with memory.
        result = copy.deepcopy(result)
        return result
