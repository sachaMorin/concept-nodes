from typing import Dict
import torch
import numpy as np

from mobile_sam import sam_model_registry
from mobile_sam import SamAutomaticMaskGenerator

from .SegmentationModel import SegmentationModel
from .utils import get_grid_coords


class AutomaticMobileSAM(SegmentationModel):
    def __init__(self, mask_generator: SamAutomaticMaskGenerator, model_type: str, checkpoint_path: str, device: str = "cuda"):
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
        masks = torch.stack([torch.from_numpy(ann["segmentation"]) for ann in anns])

        return {"masks": masks}
