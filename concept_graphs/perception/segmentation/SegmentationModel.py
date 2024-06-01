from typing import Dict

import torch
import numpy as np


class SegmentationModel:

    def __call__(self, img: np.ndarray) -> Dict[str, torch.Tensor]:
        """

        Parameters
        ----------
        img: np.ndarray
           Unprocessed uint8 RGB image of size (H, W, 3).

        Returns
        -------
        output: Dict[str, torch.Tensor]
            masks: boolean tensor of size (N, H, W) where N is the number of masks predicted.
            bbox: int tensor of size (N, 4) following the format (x_min, y_min, x_max, y_mx).
            scores: float tensor of size (N,).
            Additional keys can be added.

        """
        raise NotImplementedError
