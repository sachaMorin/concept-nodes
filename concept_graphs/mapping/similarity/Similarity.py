from typing import Tuple, List
import torch


class GeometricSimilarity:
    def __call__(
        self,
        main_pcd: torch.Tensor,
        main_centroid: torch.Tensor,
        other_pcd: torch.Tensor,
        other_centroid: torch.Tensor,
        symmetrical: bool,
    ) -> torch.Tensor:
        raise NotImplementedError


class SemanticSimilarity:
    def __call__(
        self, main_semantic: torch.Tensor, other_semantic: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class Similarity:

    def __init__(
        self,
        geometric_similarity: GeometricSimilarity,
        semantic_similarity: SemanticSimilarity,
        semantic_sim_thresh: float,
        geometric_sim_thresh: float,
    ):
        self.geometric_similarity = geometric_similarity
        self.semantic_similarity = semantic_similarity
        self.semantic_sim_thresh = semantic_sim_thresh
        self.geometric_sim_thresh = geometric_sim_thresh

    def __call__(
        self,
        main_semantic: torch.Tensor,
        main_pcd: List[torch.Tensor],
        main_centroid: torch.Tensor,
        other_semantic: torch.Tensor,
        other_pcd: List[torch.Tensor],
        other_centroid: torch.Tensor,
        mask_diagonal: bool,
    ) -> Tuple[List[bool], List[int]]:
        geometric_sim = self.geometric_similarity(
            main_pcd, main_centroid, other_pcd, other_centroid, mask_diagonal
        )
        semantic_sim = self.semantic_similarity(main_semantic, other_semantic)

        if mask_diagonal:
            semantic_sim.fill_diagonal_(-1)
            geometric_sim.fill_diagonal_(-1)

        mergeable = (geometric_sim > self.geometric_sim_thresh) & (
            semantic_sim > self.semantic_sim_thresh
        )

        semantic_sim_masked = torch.where(
            mergeable, semantic_sim, -torch.ones_like(semantic_sim)
        )

        merge = mergeable.any(dim=0).cpu().tolist()
        match_idx = semantic_sim_masked.argmax(dim=0).cpu().tolist()

        return merge, match_idx


class CombinedSimilarity:

    def __init__(
        self,
        geometric_similarity: GeometricSimilarity,
        semantic_similarity: SemanticSimilarity,
        semantic_sim_thresh: float,
        geometric_sim_thresh: float,
        sim_thresh: float,
    ):
        self.geometric_similarity = geometric_similarity
        self.semantic_similarity = semantic_similarity
        self.semantic_sim_thresh = semantic_sim_thresh
        self.geometric_sim_thresh = geometric_sim_thresh
        self.sim_thresh = sim_thresh

    def __call__(
        self,
        main_semantic: torch.Tensor,
        main_pcd: List[torch.Tensor],
        main_centroid: torch.Tensor,
        other_semantic: torch.Tensor,
        other_pcd: List[torch.Tensor],
        other_centroid: torch.Tensor,
        mask_diagonal: bool,
    ) -> Tuple[List[bool], List[int]]:
        geometric_sim = self.geometric_similarity(
            main_pcd, main_centroid, other_pcd, other_centroid, mask_diagonal
        )
        semantic_sim = self.semantic_similarity(main_semantic, other_semantic)

        if mask_diagonal:
            semantic_sim.fill_diagonal_(-1)
            geometric_sim.fill_diagonal_(-1)

        combined = (geometric_sim + semantic_sim) / 2
        mergeable = (
            (geometric_sim > self.geometric_sim_thresh)
            & (semantic_sim > self.semantic_sim_thresh)
            & (combined > self.sim_thresh)
        )

        combined_masked = torch.where(mergeable, combined, -torch.ones_like(combined))

        merge = mergeable.any(dim=0).cpu().tolist()
        match_idx = combined_masked.argmax(dim=0).cpu().tolist()

        return merge, match_idx
