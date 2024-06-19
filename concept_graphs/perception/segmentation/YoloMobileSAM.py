import os

from pathlib import Path
from typing import Tuple
import torch
import numpy as np
from ultralytics import YOLOWorld

import warnings

# Filter out user warnings from a specific package
warnings.filterwarnings("ignore", category=UserWarning, module="mobile_sam")

from mobile_sam import sam_model_registry, SamPredictor

from .SegmentationModel import SegmentationModel


class YoloMobileSAM(SegmentationModel):
    def __init__(
        self,
        yolo_checkpoint_path: str,
        yolo_class_path: str,
        yolo_device: str,
        sam_model_type: str,
        sam_checkpoint_path: str,
        sam_device: str,
        debug_images: bool,
        debug_dir: str = ".",
    ):
        self.yolo_checkpoint_path = yolo_checkpoint_path
        self.yolo_class_path = yolo_class_path
        self.yolo_device = yolo_device
        self.sam_model_type = sam_model_type
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam_device = sam_device
        self.debug_images = debug_images
        self.debug_dir = Path(debug_dir)
        self.debug_counter = 0

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

        if self.debug_images:
            os.makedirs(self.debug_dir / "detections", exist_ok=False)
            os.makedirs(self.debug_dir / "segments", exist_ok=False)

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            img_bgr = img[..., ::-1].copy()
            yolo_output = self.yolo.predict(img_bgr, verbose=False)

            bbox = yolo_output[0].boxes.xyxy.cpu().numpy()

            bbox_transformed = self.sam_predictor.transform.apply_boxes(
                bbox, original_size=img.shape[:2]
            )
            bbox_transformed = (
                torch.from_numpy(bbox_transformed).to(torch.int).to(self.sam_device)
            )

            self.sam_predictor.set_image(img)
            masks, iou_predictions, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=bbox_transformed,
                multimask_output=True,
            )

            best = torch.argmax(iou_predictions, dim=1)
            masks = masks[torch.arange(masks.size(0)), best]
            iou_predictions = iou_predictions[
                torch.arange(iou_predictions.size(0)), best
            ]
            bbox = torch.from_numpy(bbox).to(torch.int)

        if self.debug_images:
            from concept_graphs.viz.segmentation import plot_segments
            import matplotlib.pyplot as plt
            img_name = str(self.debug_counter).zfill(7) + ".png"
            yolo_output[0].plot(show=False, save=True, filename=str(self.debug_dir / "detections" / img_name))
            plot_segments(img, masks)
            plt.savefig(self.debug_dir / "segments" / img_name)
            plt.close()
            self.debug_counter += 1

        return masks, bbox, iou_predictions
