import cv2
import base64
import openai
import numpy as np
from .ImageCaptioner import ImageCaptioner
import logging

log = logging.getLogger(__name__)


class OpenAICaptioner(ImageCaptioner):
    def __init__(self, max_images: int, model: str = "gpt-4o", white_bg: bool = False):
        super().__init__(max_images)
        self.model = model
        self.client = openai.OpenAI()
        self.white_bg = white_bg
        self.role = (
            "You are a helpful assistant that describes images in a few words. "
            "You will be provided with multiple views of the same object. "
        )
        if white_bg:
            self.role += (
                "The background has been whited out to focus on the object. "
                "Some foreground objects may also have been removed. Describe the object only."
            )
        self.role += "Begin response with 'The object is'."

    def encode_images(self, images: [np.ndarray]) -> [str]:
        # To RGB
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        return [
            base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("utf-8")
            for img in images
        ]

    def postprocess_response(self, response: str) -> str:
        # Check if it begins with "The object is"
        if response.startswith("The object is"):
            response = response[13:]

        if response.endswith("."):
            response = response[:-1]

        return response

    def __call__(self, images: [np.ndarray]) -> str:
        if len(images) > self.max_images:
            images = images[: self.max_images]

        base64_images = self.encode_images(images)
        messages = [
            {"role": "system", "content": self.role},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's the object?"},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            },
                        }
                        for base64_img in base64_images
                    ],
                ],
            },
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            response = response.choices[0].message.content
        except Exception as e:
            log.warning("Error: Could not generate caption")
            response = "Invalid"

        return self.postprocess_response(response)
