from typing import Dict
import torch
import numpy as np

from mobile_sam import sam_model_registry
from mobile_sam import SamAutomaticMaskGenerator

from .SegmentationModel import SegmentationModel


def box_xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
    box_xyxy = torch.clone(box_xywh)
    box_xyxy[:, 2] = box_xyxy[:, 0] + box_xyxy[:, 2]
    box_xyxy[:, 3] = box_xyxy[:, 1] + box_xyxy[:, 3]
    return box_xyxy


class AutomaticMobileSAM(SegmentationModel):
    def __init__(self, mask_generator: SamAutomaticMaskGenerator, model_type: str, checkpoint_path: str,
                 device: str = "cuda"):
        """Mobile-SAM model with grid-based prompting."""
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device

        mobile_sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        mobile_sam.to(device=self.device)
        mobile_sam.eval()

        self.predictor = mask_generator(model=mobile_sam)

    def __call__(self, img: np.ndarray) -> Dict[str, torch.Tensor]:
        anns = self.predictor.generate(img)
        masks, bbox, score = [], [], []

        for ann in anns:
            masks.append(torch.from_numpy(ann["segmentation"]))
            bbox.append(torch.tensor(ann["bbox"]))
            score.append(ann["predicted_iou"])

        result = {"masks": torch.stack(masks), "bbox": torch.stack(bbox), "scores": torch.tensor(score)}

        result["bbox"] = box_xywh_to_xyxy(result["bbox"])

        return result
