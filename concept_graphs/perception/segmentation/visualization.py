from typing import List, Tuple, Union
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt


def plot_grid_images(images: List[Union[np.ndarray, torch.Tensor]], grid_width: int = 4) -> None:
    n_images = len(images)
    grid_height = int(np.ceil(n_images / grid_width))

    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width * 4, grid_height * 4))

    for i, ax in enumerate(axs.flat):
        img = images[i]
        if type(img) == torch.Tensor:
            img = F.to_pil_image(img)

        if i < n_images:
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()


def plot_segments(image: Union[np.ndarray, torch.Tensor], masks: torch.Tensor) -> None:
    if type(image) is not torch.Tensor:
        image = torchvision.transforms.ToTensor()(image)
    random_colors = [(r, g, b) for r, g, b in (255 * torch.rand((masks.shape[0], 3))).int()]
    img_with_segmentations = torchvision.utils.draw_segmentation_masks(image, masks, colors=random_colors)
    plt.imshow(F.to_pil_image(img_with_segmentations))
