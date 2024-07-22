from typing import List
import numpy as np
import logging


# A logger for this file
log = logging.getLogger(__name__)


class ImageCaptioner:
    def __init__(self, max_images: int):
        self.max_images = (
            max_images  # Subsample object views if too many images are provided
        )

    def __call__(self, List: [np.ndarray]) -> str:
        raise NotImplementedError

    def caption_map(self, map: "ObjectMap") -> None:
        for obj in map:
            views = [v.rgb for v in obj.segments.get_sorted()]
            obj.caption = self(views)
            log.info(obj.caption)
