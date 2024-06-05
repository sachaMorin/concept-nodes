from typing import Tuple
import torch
import numpy as np
from ultralytics import YOLOWorld

import warnings

# Filter out user warnings from a specific package
warnings.filterwarnings("ignore", category=UserWarning, module="mobile_sam")

from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.amg import batched_mask_to_box
from torchvision.ops.boxes import batched_nms

from .SegmentationModel import SegmentationModel
from .utils import get_grid_coords, bbox_area


class YoloMobileSAM(SegmentationModel):
    def __init__(
            self,
            yolo_checkpoint_path: str,
            yolo_class_path: str,
            yolo_device: str,
            sam_model_type: str,
            sam_checkpoint_path: str,
            sam_device: str,
    ):
        self.yolo_checkpoint_path = yolo_checkpoint_path
        self.yolo_class_path = yolo_class_path
        self.yolo_device = yolo_device
        self.sam_model_type = sam_model_type
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam_device = sam_device

        # YOLO
        with open(self.yolo_class_path, "r") as f:
            self.classes = f.read().splitlines()
        self.yolo = YOLOWorld(self.yolo_checkpoint_path, verbose=False)
        self.yolo.set_classes(self.classes)
        self.yolo.to(self.yolo_device)

        # Mobile SAM
        mobile_sam = sam_model_registry[self.sam_model_type](
            checkpoint=self.sam_checkpoint_path
        )
        mobile_sam.to(device=self.sam_device)
        mobile_sam.eval()

        self.sam_predictor = SamPredictor(mobile_sam)

    def __call__(
            self, img: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            img_bgr = img[..., ::-1].copy()
            yolo_output = self.yolo.predict(img_bgr, verbose=False)

            bbox = yolo_output[0].boxes.xyxy.cpu().numpy()

            bbox_transformed = self.sam_predictor.transform.apply_boxes(bbox, original_size=img.shape[:2])
            bbox_transformed = torch.from_numpy(bbox_transformed).to(torch.int).to(self.sam_device)

            self.sam_predictor.set_image(img)
            masks, iou_predictions, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=bbox_transformed,
                multimask_output=True,
            )

            best = torch.argmax(iou_predictions, dim=1)
            masks = masks[torch.arange(masks.size(0)), best]
            iou_predictions = iou_predictions[torch.arange(iou_predictions.size(0)), best]
            bbox = torch.from_numpy(bbox).to(torch.int)

        # from concept_graphs.viz.segmentation import plot_segments
        # import matplotlib.pyplot as plt
        # yolo_output[0].show()
        # plot_segments(img, masks)
        # plt.show()
        # exit()

        return masks, bbox, iou_predictions
