from typing import List
import numpy as np

class ImageCaptioner:
    def __init__(self, max_images: int):
        self.max_images = max_images # Subsample object views if too many images are provided
    def __call__(self, List: [np.ndarray]) -> str:
        raise NotImplementedError