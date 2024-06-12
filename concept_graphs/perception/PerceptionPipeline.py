from typing import Dict, List, Union
import numpy as np
from .ft_extraction.FeatureExtractor import FeatureExtractor
from .segmentation.SegmentationModel import SegmentationModel
from .rgbd_to_pcd import rgbd_to_object_pcd
from .segmentation.utils import (
    extract_rgb_crops,
    extract_mask_crops,
    safe_bbox_inflate,
    mask_subtract_contained,
)


class PerceptionPipeline:
    def __init__(
        self,
        segmentation_model: SegmentationModel,
        ft_extractor: FeatureExtractor,
        inflate_bbox_px: int,
        depth_trunc: float,
        mask_subtract_contained: bool,
        semantic_similarity: "SemanticSimilarity" = None,
        bg_classes: Union[List[str], None] = None,
        bg_sim_thresh: float = 1.0,
        crop_bg_color: Union[int, None] = None,
        min_mask_area_px: int = 25,
    ):
        self.segmentation_model = segmentation_model
        self.ft_extractor = ft_extractor
        self.depth_trunc = depth_trunc
        self.inflate_bbox_px = inflate_bbox_px
        self.crop_bg_color = crop_bg_color
        self.mask_subtract_contained = mask_subtract_contained
        self.semantic_similarity = semantic_similarity
        self.bg_classes = bg_classes
        self.bg_sim_thresh = bg_sim_thresh
        self.min_mask_area_px = min_mask_area_px

        self.bg_features = None

        # Compute features for bg_class
        if self.bg_classes is not None:
            self.bg_features = self.ft_extractor.encode_text(self.bg_classes)
            self.bg_features.to(self.ft_extractor.device)

    def __call__(
        self, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        masks, bbox, scores = self.segmentation_model(rgb)

        if self.inflate_bbox_px > 0:
            bbox = safe_bbox_inflate(
                bbox, self.inflate_bbox_px, rgb.shape[1], rgb.shape[0]
            )

        masks, bbox, scores = (
            masks.cpu().numpy(),
            bbox.cpu().numpy(),
            scores.cpu().numpy(),
        )

        if self.mask_subtract_contained:
            masks = mask_subtract_contained(bbox, masks)

        # Segment filtering
        areas = masks.sum(axis=-1).sum(axis=-1)
        keep = areas > self.min_mask_area_px

        # Use areas as scores
        masks, bbox, scores = masks[keep], bbox[keep], areas[keep]

        mask_crops = extract_mask_crops(masks, bbox)
        rgb_crops = extract_rgb_crops(rgb, bbox, mask_crops, self.crop_bg_color)

        features = self.ft_extractor(rgb_crops)

        if self.bg_features is not None:
            sim = self.semantic_similarity(features, self.bg_features)
            bg = (sim > self.bg_sim_thresh).any(dim=1).cpu().numpy()
        else:
            bg = np.zeros(len(features), dtype=bool)

        features = features.cpu().numpy()

        pcd_points, pcd_rgb = rgbd_to_object_pcd(
            rgb, depth, masks, intrinsics, depth_trunc=self.depth_trunc
        )

        return dict(
            rgb_crops=rgb_crops,
            mask_crops=mask_crops,
            features=features,
            pcd_points=pcd_points,
            pcd_rgb=pcd_rgb,
            scores=scores,
            is_bg=bg,
        )
