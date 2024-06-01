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
        masks: Dict[str, torch.Tensor]
            The "masks" key should be a boolean tensor of size (N, H, W) where N is the number of masks predicted.
            Additional keys can be added.

        """
        raise NotImplementedError
