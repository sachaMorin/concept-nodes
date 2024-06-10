from typing import Tuple
import torch
import numpy as np

import warnings

# Filter out user warnings from a specific package
warnings.filterwarnings("ignore", category=UserWarning, module="mobile_sam")

from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.amg import batched_mask_to_box
from torchvision.ops.boxes import batched_nms

from .SegmentationModel import SegmentationModel
from .utils import get_grid_coords, bbox_area


class GridMobileSAM(SegmentationModel):
    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        grid_jitter: bool,
        model_type: str,
        checkpoint_path: str,
        nms_iou_threshold: float,
        device: str,
        min_mask_area_px: int,
        min_bbox_side_px: int,
    ):
        """Mobile-SAM model with grid-based prompting."""
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid_jitter = grid_jitter
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.nms_iou_threshold = nms_iou_threshold
        self.min_mask_area_px = min_mask_area_px
        self.min_bbox_side_px = min_bbox_side_px

        mobile_sam = sam_model_registry[self.model_type](
            checkpoint=self.checkpoint_path
        )
        mobile_sam.to(device=self.device)
        mobile_sam.eval()

        self.predictor = SamPredictor(mobile_sam)
        sz = mobile_sam.image_encoder.img_size
        self.grid_coords = (
            get_grid_coords(
                grid_width,
                grid_height,
                sz,
                sz,
                self.device,
                jitter=self.grid_jitter,
                uniform_jitter=False,
            )
            .cpu()
            .numpy()
        )
        self.grid_coords = self.predictor.transform.apply_coords(
            self.grid_coords, original_size=(sz, sz)
        )
        self.grid_coords = (
            torch.from_numpy(self.grid_coords).to(self.device).unsqueeze(1)
        )
        self.labels = torch.ones(grid_width * grid_height, 1).to(self.device)

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.predictor.set_image(img)
        masks, iou_predictions, _ = self.predictor.predict_torch(
            point_coords=self.grid_coords,
            point_labels=self.labels,
            multimask_output=True,
        )

        best = torch.argmax(iou_predictions, dim=1)
        masks = masks[torch.arange(masks.size(0)), best]
        iou_predictions = iou_predictions[torch.arange(iou_predictions.size(0)), best]
        bbox = batched_mask_to_box(masks)

        # Segment filtering
        areas = masks.sum(dim=[1, 2])
        bbox_ok = (bbox[:, 2] - bbox[:, 0] > self.min_bbox_side_px) & (
            bbox[:, 3] - bbox[:, 1] > self.min_bbox_side_px
        )
        keep = (areas > self.min_mask_area_px) & bbox_ok
        masks, bbox, iou_predictions = masks[keep], bbox[keep], iou_predictions[keep]

        # Nms
        keep = batched_nms(
            bbox.to(torch.float),
            iou_predictions,
            torch.zeros_like(iou_predictions),
            self.nms_iou_threshold,
        )
        masks, bbox, iou_predictions = masks[keep], bbox[keep], iou_predictions[keep]

        return masks, bbox, iou_predictions
