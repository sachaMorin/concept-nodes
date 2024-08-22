from typing import Tuple, List
import torch
from .Similarity import GeometricSimilarity


def point_cloud_overlap(
    pcd_1: torch.Tensor, pcd_2: torch.Tensor, eps: float
) -> Tuple[torch.Tensor]:
    pcd_1 = pcd_1.unsqueeze(1)  # (n1, 1, 3)
    pcd_2 = pcd_2.unsqueeze(0)  # (1, n2, 3)

    dist = ((pcd_1 - pcd_2) ** 2).sum(dim=-1)  # (n1, n2)

    is_close = torch.sqrt(dist) < eps

    d1 = is_close.any(dim=1).to(torch.float).mean()
    d2 = is_close.any(dim=0).to(torch.float).mean()

    return d1, d2


class PointCloudOverlapClosestK(GeometricSimilarity):
    """Point Cloud Overlap with closest k other point clouds in terms of centroid distance."""

    def __init__(self, eps: float, agg: str, k: int = 3, max_dist_centroid=10.0):
        self.eps = eps
        self.agg = agg
        self.k = max(2, k)
        self.max_dist_centroid = max_dist_centroid

    def __call__(
        self,
        main_pcd: List[torch.Tensor],
        main_centroid: torch.Tensor,
        other_pcd: List[torch.Tensor],
        other_centroid: torch.Tensor,
        is_symmetrical: bool,
    ) -> torch.Tensor:
        dist_centroids = torch.cdist(main_centroid, other_centroid)

        k = min(len(main_pcd), self.k)
        closest_k = torch.topk(dist_centroids, k=k, dim=0, largest=False).indices.T
        result = torch.zeros_like(dist_centroids)

        for other_i, other_pcd_i in enumerate(other_pcd):
            indices = closest_k[other_i]
            for main_i in indices:
                if is_symmetrical and main_i == other_i:
                    result[main_i, other_i] = 1.0
                    continue

                if dist_centroids[main_i, other_i] > self.max_dist_centroid:
                    continue

                sim1, sim2 = point_cloud_overlap(
                    main_pcd[main_i], other_pcd_i, eps=self.eps
                )

                if self.agg == "sum":
                    sim = sim1 + sim2
                elif self.agg == "mean":
                    sim = (sim1 + sim2) / 2
                elif self.agg == "max":
                    sim = torch.max(sim1, sim2)
                elif self.agg == "other":
                    sim = sim2
                else:
                    raise ValueError(f"Unknown aggregation method {self.agg}")

                result[main_i, other_i] = sim

        return result


class PointCloudOverlap(GeometricSimilarity):
    """Point Cloud Overlap."""

    def __init__(self, eps: float, agg: str):
        self.eps = eps
        self.agg = agg

    def __call__(
        self,
        main_pcd: List[torch.Tensor],
        main_centroid: torch.Tensor,
        other_pcd: List[torch.Tensor],
        other_centroid: torch.Tensor,
        is_symmetrical: bool,
    ) -> torch.Tensor:

        result = torch.zeros(len(main_pcd), len(other_pcd), device=main_centroid.device)
        for main_i, main_pcd_i in enumerate(main_pcd):
            for other_i, other_pcd_i in enumerate(other_pcd):
                if is_symmetrical and main_i == other_i:
                    result[main_i, other_i] = 1.0
                    continue

                sim1, sim2 = point_cloud_overlap(main_pcd_i, other_pcd_i, eps=self.eps)

                if self.agg == "sum":
                    sim = sim1 + sim2
                elif self.agg == "mean":
                    sim = (sim1 + sim2) / 2
                elif self.agg == "max":
                    sim = torch.max(sim1, sim2)
                elif self.agg == "other":
                    sim = sim2
                else:
                    raise ValueError(f"Unknown aggregation method {self.agg}")

                result[main_i, other_i] = sim

        # torch.set_printoptions(precision=2)
        # print(result)

        return result
