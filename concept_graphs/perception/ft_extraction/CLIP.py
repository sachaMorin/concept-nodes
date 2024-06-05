from typing import List
import torch
import open_clip
import numpy as np
from PIL import Image

from .FeatureExtractor import FeatureExtractor


class CLIP(FeatureExtractor):
    def __init__(self, **kwargs):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(**kwargs)
        self.model.eval()
        self.device = kwargs["device"]
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(kwargs["model_name"])

    def __call__(self, imgs: List[np.ndarray]) -> torch.Tensor:
        with torch.no_grad():
            imgs_processed = [self.preprocess(Image.fromarray(img)) for img in imgs]
            batch = torch.stack(imgs_processed).to(self.device)
            features = self.model.encode_image(batch)
            features /= features.norm(dim=-1, keepdim=True)

        return features

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            text_tokens = self.tokenizer(texts).to(self.device)
            features = self.model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return features
