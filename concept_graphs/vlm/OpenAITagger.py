import cv2
import base64
import openai
import numpy as np
from .ImageCaptioner import ImageCaptioner
import logging
from .OpenAICaptioner import OpenAICaptioner

log = logging.getLogger(__name__)


class OpenAITagger(OpenAICaptioner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefix = "The best word to describe this object is:"
        self.role = (
            "You are a helpful assistant that describes objects with a single word. "
            "You will be provided with multiple views of the same object. Provide the best word describing the central object in the views."
        )
        if self.white_bg:
            self.role += (
                "The background has been whited out to focus on the object. "
                "Some foreground objects may also have been removed. Describe the object only."
            )
        self.role += f"Begin response with '{self.prefix}'"

    def postprocess_response(self, response: str) -> str:
        response = response.replace(self.prefix, "")
        response = response.replace("*", "")
        response = response.replace("'", "")
        response = response.replace('"', "")
        response = response.replace(".", "")
        response = response.replace(",", "")

        return response

    def caption_map(self, map: "ObjectMap") -> None:
        for obj in map:
            views = [v.rgb for v in obj.segments]
            obj.tag = self(views)
            log.info(obj.tag)
