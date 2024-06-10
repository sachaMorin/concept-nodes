import cv2
import base64
import openai
import numpy as np
from .ImageCaptioner import ImageCaptioner
import logging

log = logging.getLogger(__name__)


class OpenAICaptioner(ImageCaptioner):
    def __init__(self, max_images: int, model: str = "gpt-4o"):
        super().__init__(max_images)
        self.model = model
        self.client = openai.OpenAI()
        self.role = (
            "You are a helpful assistant that describes images in a few words. "
            "You will be provided with multiple views of the same object. "
            "The background has been whited out to focus on the object. "
            "Some foreground objects may also have been removed. Describe the object only."
            "Begin response with 'The object is'."
        )

    def encode_images(self, images: [np.ndarray]) -> [str]:
        # To RGB
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        return [
            base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("utf-8")
            for img in images
        ]

    def __call__(self, images: [np.ndarray]) -> str:
        if len(images) > self.max_images:
            # Subsample images
            indices = np.random.choice(len(images), self.max_images, replace=False)
            images = [images[i] for i in indices]

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

        # Check if it begins with "The object is"
        if response.startswith("The object is"):
            response = response[13:]

        if response.endswith("."):
            response = response[:-1]

        return response
