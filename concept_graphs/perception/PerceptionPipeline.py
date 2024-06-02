from typing import Dict, List, Union
import numpy as np
from .ft_extraction.FeatureExtractor import FeatureExtractor
from .segmentation.SegmentationModel import SegmentationModel
from .rgbd_to_pcd import rgbd_to_object_pcd
from .segmentation.utils import extract_rgb_crops, extract_mask_crops


class PerceptionPipeline:
    def __init__(self, segmentation_model: SegmentationModel, ft_extractor: FeatureExtractor, depth_trunc: float = 8.0):
        self.segmentation_model = segmentation_model
        self.ft_extractor = ft_extractor
        self.depth_trunc = depth_trunc

    def __call__(self, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        masks, bbox, scores = self.segmentation_model(rgb)
        masks, bbox, scores = masks.cpu().numpy(), bbox.cpu().numpy(), scores.cpu().numpy()

        rgb_crops = extract_rgb_crops(rgb, bbox)
        mask_crops = extract_mask_crops(masks, bbox)

        features = self.ft_extractor(rgb_crops)
        features = features.cpu().numpy()

        pcd_points, pcd_rgb = rgbd_to_object_pcd(rgb, depth, masks, intrinsics, depth_trunc=self.depth_trunc)

        return dict(
            bbox=bbox,
            scores=scores,
            rgb_crops=rgb_crops,
            mask_crops=mask_crops,
            features=features,
            pcd_points=pcd_points,
            pcd_rgb=pcd_rgb,
        )
