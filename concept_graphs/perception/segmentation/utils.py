from typing import List, Union
import numpy as np
import torch


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
        crop = img[box[1] : box[3], box[0] : box[2]]
        if mask_crops is not None and bg_color is not None:
            crop = np.where(mask_crops[i][:, :, np.newaxis], crop, bg_color)
        crops.append(crop)
    return crops


def extract_mask_crops(masks: np.ndarray, bbox: np.ndarray) -> List[np.ndarray]:
    crops = []
    for mask, box in zip(masks, bbox):
        crop = mask[box[1] : box[3], box[0] : box[2]]
        crops.append(crop)
    return crops


def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    """
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.

    Stolen from original ConceptGraphs repo at https://github.com/concept-graphs/concept-graphs

    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2

    Returns:
        mask_sub: (N, H, W), binary mask
    """
    N = xyxy.shape[0]  # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(
        xyxy[:, None, 2:], xyxy[None, :, 2:]
    )  # right-bottom points (N, N, 2)

    inter = (rb - lt).clip(
        min=0
    )  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T  # (N, N)

    # if the intersection area is smaller than th2 of the area of box1,
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
    contained_idx = contained.nonzero()  # (num_contained, 2)

    mask_sub = mask.clone()  # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (
            ~mask_sub[contained_idx[1][i]]
        )

    return mask_sub
