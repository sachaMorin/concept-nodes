from typing import Tuple
import torch
from .Similarity import GeometricSimilarity


class CentroidDist(GeometricSimilarity):
    def __call__(
        self, main_centroid: torch.Tensor, other_centroid: torch.Tensor
    ) -> torch.Tensor:
        dist = torch.norm(main_centroid - other_centroid, dim=1)
        return 1 / (dist + 1e-6)


class ChamferDist(GeometricSimilarity):
    def __init__(self, agg: str):
        self.agg = agg

    def __call__(
        self, main_geometry: torch.Tensor, other_geometry: torch.Tensor
    ) -> torch.Tensor:
        if main_geometry.ndim == 2:
            main_geometry = main_geometry.unsqueeze(0)
        if other_geometry.ndim == 2:
            other_geometry = other_geometry.unsqueeze(1)

        main_geometry = main_geometry.unsqueeze(1)
        main_geometry = main_geometry.unsqueeze(3)
        other_geometry = other_geometry.unsqueeze(0)
        other_geometry = other_geometry.unsqueeze(2)

        dist = ((main_geometry - other_geometry) ** 2).sum(dim=-1)
        d1 = dist.min(dim=3)[0].mean(dim=2)
        d2 = dist.min(dim=2)[0].mean(dim=2)

        if self.agg == "sum":
            result = d1 + d2
        elif self.agg == "min":
            result = 2 * torch.min(d1, d2)  # Same scale as the sum
        else:
            raise ValueError(f"Unknown aggregation method {self.agg}")

        return 1 / (result + 1e-6)


def radius_overlap(pcds_1: torch.Tensor, pcds_2: torch.Tensor, eps: float) -> Tuple[torch.Tensor]:
        pcds_1 = pcds_1.unsqueeze(1)
        pcds_1 = pcds_1.unsqueeze(3)
        pcds_2 = pcds_2.unsqueeze(0)
        pcds_2 = pcds_2.unsqueeze(2)

        dist = ((pcds_1 - pcds_2) ** 2).sum(dim=-1)

        is_close = torch.sqrt(dist) < eps
        d1 = is_close.any(dim=3).to(torch.float).mean(dim=2)
        d2 = is_close.any(dim=2).to(torch.float).mean(dim=2)

        return d1, d2


class RadiusOverlap(GeometricSimilarity):
    def __init__(self, eps: float, agg: str, batch_size: int = 20):
        self.eps = eps
        self.agg = agg
        self.batch_size = batch_size

    def __call__(
        self, main_geometry: torch.Tensor, other_geometry: torch.Tensor
    ) -> torch.Tensor:
        if main_geometry.ndim == 2:
            main_geometry = main_geometry.unsqueeze(0)
        if other_geometry.ndim == 2:
            other_geometry = other_geometry.unsqueeze(0)

        d1_buffer, d2_buffer = list(), list()

        n_batches = main_geometry.shape[0] // self.batch_size + 1 if self.batch_size > 0 else 1

        for batch_main_geometry in torch.tensor_split(main_geometry, n_batches, dim=0):
            d1, d2 = radius_overlap(batch_main_geometry, other_geometry, self.eps)
            d1_buffer.append(d1)
            d2_buffer.append(d2)

        d1 = torch.concat(d1_buffer, dim=0)
        d2 = torch.concat(d2_buffer, dim=0)

        if self.agg == "sum":
            result = d1 + d2
        elif self.agg == "mean":
            result = (d1 + d2) / 2
        elif self.agg == "max":
            result = torch.max(d1, d2)
        else:
            raise ValueError(f"Unknown aggregation method {self.agg}")

        return result

