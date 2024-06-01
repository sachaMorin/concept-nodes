from typing import List

import torch
import numpy as np


class FeatureExtractor:

    def __call__(self, imgs: List[np.ndarray]) -> torch.Tensor:
        """

        Parameters
        ----------
        imgs: List[np.ndarray]
           List of N unprocessed uint8 RGB images.

        Returns
        -------
        output: torch.Tensor
            Tensor of size (N, K) with image features.
        """
        raise NotImplementedError
