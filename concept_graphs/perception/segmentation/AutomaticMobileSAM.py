from typing import Tuple
import torch
import numpy as np

import warnings

# Filter out user warnings from a specific package
warnings.filterwarnings("ignore", category=UserWarning, module="mobile_sam")

from mobile_sam import sam_model_registry
from mobile_sam import SamAutomaticMaskGenerator

from .SegmentationModel import SegmentationModel


def box_xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
    box_xyxy = torch.clone(box_xywh)
    box_xyxy[:, 2] = box_xyxy[:, 0] + box_xyxy[:, 2]
    box_xyxy[:, 3] = box_xyxy[:, 1] + box_xyxy[:, 3]
    return box_xyxy


class AutomaticMobileSAM(SegmentationModel):
    def __init__(
        self,
        mask_generator: SamAutomaticMaskGenerator,
        model_type: str,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        """Mobile-SAM model with grid-based prompting."""
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device

        mobile_sam = sam_model_registry[self.model_type](
            checkpoint=self.checkpoint_path
        )
        mobile_sam.to(device=self.device)
        mobile_sam.eval()

        self.predictor = mask_generator(model=mobile_sam)

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anns = self.predictor.generate(img)
        masks, bbox, score = [], [], []

        for ann in anns:
            masks.append(torch.from_numpy(ann["segmentation"]))
            bbox.append(torch.tensor(ann["bbox"]))
            score.append(ann["predicted_iou"])

        masks, bbox, score = torch.stack(masks), torch.stack(bbox), torch.tensor(score)
        bbox = box_xywh_to_xyxy(bbox)

        return masks, bbox, score
