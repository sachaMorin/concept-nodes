from typing import Dict
import torch
import numpy as np

from mobile_sam import sam_model_registry, SamPredictor

from .SegmentationModel import SegmentationModel
from .utils import get_grid_coords


class GridMobileSAM(SegmentationModel):
    def __init__(self, grid_width: int, grid_height: int, model_type: str, checkpoint_path: str, device: str = "cuda"):
        """Mobile-SAM model with grid-based prompting."""
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.grid_width = grid_width
        self.grid_height = grid_height

        mobile_sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        mobile_sam.to(device=self.device)
        mobile_sam.eval()

        self.predictor = SamPredictor(mobile_sam)
        inference_image_size = mobile_sam.image_encoder.img_size
        self.grid_coords = get_grid_coords(grid_width, grid_height, inference_image_size,
                                           inference_image_size, self.device).unsqueeze(1)
        self.labels = torch.ones(grid_width * grid_height, 1).to(self.device)

    def __call__(self, img: np.ndarray) -> Dict[str, torch.Tensor]:
        self.predictor.set_image(img)
        masks, iou_predictions, _ = self.predictor.predict_torch(point_coords=self.grid_coords,
                                                                 point_labels=self.labels,
                                                                 multimask_output=True)

        best = torch.argmax(iou_predictions, dim=1)
        masks = masks[torch.arange(masks.size(0)), best]
        iou_predictions = iou_predictions[torch.arange(iou_predictions.size(0)), best]

        return {"masks": masks, "iou_predictions": iou_predictions}
