from typing import List, Union
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .utils import similarities_to_rgb


def plot_grid_images(images: List[Union[np.ndarray, torch.Tensor]], masks: List[np.ndarray] = None, grid_width: int = 4) -> None:
    n_images = len(images)
    grid_height = int(np.ceil(n_images / grid_width))
    mask_color = (np.random.rand(3) * 255).astype(np.uint8).reshape((1, 3))

    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width * 4, grid_height * 4))


    for i, ax in enumerate(axs.flat):
        if i < n_images:
            img = images[i]
            if type(img) == torch.Tensor:
                img = F.to_pil_image(img)
            ax.imshow(img)
            if masks is not None:
                mask = masks[i]
                mask = mask[:, :, np.newaxis] * mask_color
                ax.imshow(mask, alpha=0.5)
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()


def plot_segments(image: np.ndarray, masks: torch.Tensor) -> None:
    image = torchvision.transforms.ToTensor()(image)
    np.random.seed(42)
    random_colors = [(r, g, b) for r, g, b in (255 * torch.rand((masks.shape[0], 3))).int()]
    img_with_segmentations = torchvision.utils.draw_segmentation_masks(image, masks, colors=random_colors)
    plt.axis("off")
    plt.imshow(F.to_pil_image(img_with_segmentations))

def plot_segments_similarity(image: np.ndarray, masks: torch.Tensor, similarities: np.ndarray) -> None:
    image = torchvision.transforms.ToTensor()(image)
    colors = similarities_to_rgb(similarities, "viridis")
    img_with_segmentations = torchvision.utils.draw_segmentation_masks(image, masks, colors=colors)
    plt.axis("off")
    plt.imshow(F.to_pil_image(img_with_segmentations))


def plot_bbox(image: np.ndarray, bbox: torch.Tensor) -> None:
    image = (255 * torchvision.transforms.ToTensor()(image)).to(torch.uint8)
    np.random.seed(42)
    random_colors = [(r, g, b) for r, g, b in (255 * torch.rand((bbox.shape[0], 3))).int()]
    img_with_bbox = torchvision.utils.draw_bounding_boxes(image, bbox, colors=random_colors, width=3)
    plt.axis("off")
    plt.imshow(F.to_pil_image(img_with_bbox))
