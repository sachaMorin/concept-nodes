from typing import List, Union
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)

def get_grid_coords(
        grid_width: int,
        grid_height: int,
        image_width: int,
        image_height: int,
        device: str,
        jitter: bool = False,
        uniform_jitter: bool = True,
) -> torch.Tensor:
    y = torch.linspace(0, 1, grid_height + 2, device=device)[1:-1] * image_height
    x = torch.linspace(0, 1, grid_width + 2, device=device)[1:-1] * image_width
    x = x.int()
    y = y.int()
    grid = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape((-1, 2))

    if jitter:
        step_x = image_width // grid_width
        step_y = image_height // grid_height

        if uniform_jitter:
            noise_x = torch.randint(
                -step_x // 2, step_x // 2, (1,), device=device
            ).repeat(grid.shape[0])
            noise_y = torch.randint(
                -step_y // 2, step_y // 2, (1,), device=device
            ).repeat(grid.shape[0])
        else:
            noise_x = torch.randint(
                -step_x // 2, step_x // 2, (grid.shape[0],), device=device
            )
            noise_y = torch.randint(
                -step_y // 2, step_y // 2, (grid.shape[0],), device=device
            )

        grid[:, 0] += noise_x
        grid[:, 1] += noise_y

    return grid


def bbox_area(bbox: torch.Tensor) -> torch.Tensor:
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def safe_bbox_inflate(
        bbox: torch.Tensor, inflate_px: int, img_width: int, img_height: int
) -> torch.Tensor:
    bbox = bbox.clone()
    bbox[:, 0] = torch.clamp(bbox[:, 0] - inflate_px, 0, img_width - 1)
    bbox[:, 1] = torch.clamp(bbox[:, 1] - inflate_px, 0, img_height - 1)
    bbox[:, 2] = torch.clamp(bbox[:, 2] + inflate_px, 0, img_width - 1)
    bbox[:, 3] = torch.clamp(bbox[:, 3] + inflate_px, 0, img_height - 1)
    return bbox


def extract_rgb_crops(
        img: np.ndarray,
        bbox: np.ndarray,
        mask_crops: Union[List[np.ndarray], None] = None,
        bg_color: Union[int, None] = None,
) -> List[np.ndarray]:
    crops = []
    for i, box in enumerate(bbox):
        crop = img[box[1]: box[3], box[0]: box[2]]
        if mask_crops is not None and bg_color is not None:
            crop = np.where(mask_crops[i][:, :, np.newaxis], crop, bg_color)
        crops.append(crop)
    return crops


def extract_mask_crops(masks: np.ndarray, bbox: np.ndarray) -> List[np.ndarray]:
    crops = []
    for mask, box in zip(masks, bbox):
        crop = mask[box[1]: box[3], box[0]: box[2]]
        crops.append(crop)
    return crops


def bbox_overlap(xyxy: torch.Tensor) -> torch.Tensor:
    # Compute intersection boxes
    lt = torch.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = torch.minimum(
        xyxy[:, None, 2:], xyxy[None, :, 2:]
    )  # right-bottom points (N, N, 2)

    inter = (rb - lt).clip(
        min=0
    )  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    return inter_areas


def mask_subtract_contained(xyxy: torch.Tensor, mask: torch.Tensor):
    """Adapted from the original CG repo at https://github.com/concept-graphs/concept-graphs"""
    overlap = bbox_overlap(xyxy)
    areas = mask.sum(dim=(1, 2))

    overlap = torch.tril(overlap, diagonal=-1)

    is_overlapping = overlap > 0
    is_overlapping_idx = zip(*is_overlapping.nonzero(as_tuple=True))

    mask_sub = mask.clone()
    for i, j in is_overlapping_idx:
        if areas[i] > areas[j]:
            mask_sub[i] = mask_sub[i] & (~mask[j])
        else:
            mask_sub[j] = mask_sub[j] & (~mask[i])

    still_overlaps = mask_sub.sum(dim=0) > 1

    if torch.any(still_overlaps):
        logging.warning(f"Some segments still overlaps: {still_overlaps.sum()} px")

    return mask_sub