from typing import List
import numpy as np
import torch


def get_grid_coords(grid_width: int, grid_height: int, image_width: int, image_height: int,
                    device: str, jitter: bool = False, uniform_jitter: bool = True) -> torch.Tensor:
    y = torch.linspace(0, 1, grid_height + 2, device=device)[1:-1] * image_height
    x = torch.linspace(0, 1, grid_width + 2, device=device)[1:-1] * image_width
    x = x.int()
    y = y.int()
    grid = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape((-1, 2))

    if jitter:
        step_x = image_width // grid_width
        step_y = image_height // grid_height

        if uniform_jitter:
            noise_x = torch.randint(-step_x // 2, step_x // 2, (1,), device=device).repeat(grid.shape[0])
            noise_y = torch.randint(-step_y // 2, step_y // 2, (1,), device=device).repeat(grid.shape[0])
        else:
            noise_x = torch.randint(-step_x // 2, step_x // 2, (grid.shape[0],), device=device)
            noise_y = torch.randint(-step_y // 2, step_y // 2, (grid.shape[0],), device=device)

        grid[:, 0] += noise_x
        grid[:, 1] += noise_y

    return grid


def bbox_area(bbox: torch.Tensor) -> torch.Tensor:
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def extract_rgb_crops(img: np.ndarray, bbox: np.ndarray) -> List[np.ndarray]:
    crops = []
    for box in bbox:
        crop = img[box[1]:box[3], box[0]:box[2]]
        crops.append(crop)
    return crops


def extract_mask_crops(masks: np.ndarray, bbox: np.ndarray) -> List[np.ndarray]:
    crops = []
    for mask, box in zip(masks, bbox):
        crop = mask[box[1]:box[3], box[0]:box[2]]
        crops.append(crop)
    return crops
