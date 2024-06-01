from typing import List
import numpy as np
import torch


def get_grid_coords(grid_width: int, grid_height: int, image_width: int, image_height: int,
                    device: str) -> torch.Tensor:
    x = torch.linspace(0, 1, grid_width + 1, device=device)[1:] * image_height
    y = torch.linspace(0, 1, grid_height + 1, device=device)[1:] * image_width
    x = x.int()
    y = y.int()
    grid = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape((-1, 2))
    return grid


def bbox_area(bbox: torch.Tensor) -> torch.Tensor:
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def extract_crops(img: np.ndarray, bbox: torch.Tensor) -> List[np.ndarray]:
    crops = []
    for box in bbox:
        crop = img[box[1]:box[3], box[0]:box[2]]
        crops.append(crop)
    return crops
