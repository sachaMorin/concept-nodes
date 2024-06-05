from typing import Tuple, List
import torch


def match_similarities(
    main_semantic: torch.Tensor,
    main_geometry: torch.Tensor,
    other_semantic: torch.Tensor,
    other_geometry: torch.Tensor,
    semantic_sim_thresh: float,
    geometric_sim_thresh: float,
    geometry_mode: str = "centroid",
    mask_diagonal: bool = True,
) -> Tuple[List[bool], List[int]]:
    semantic_sim = other_semantic @ main_semantic.t()

    if geometry_mode == "centroid":
        geometric_sim = 1 / (torch.cdist(other_geometry, main_geometry) + 1e-6)
    else:
        raise ValueError(f"Unknown geometry mode: {geometry_mode}")

    if mask_diagonal:
        semantic_sim.fill_diagonal_(-1)
        geometric_sim.fill_diagonal_(0)

    mergeable = (geometric_sim > geometric_sim_thresh) & (
        semantic_sim > semantic_sim_thresh
    )

    semantic_sim_masked = torch.where(
        mergeable, semantic_sim, -torch.ones_like(semantic_sim)
    )

    merge = mergeable.any(dim=1).cpu().tolist()
    match_idx = semantic_sim_masked.argmax(dim=1).cpu().tolist()

    return merge, match_idx
