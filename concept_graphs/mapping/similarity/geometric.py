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


class RadiusOverlap(GeometricSimilarity):
    def __init__(self, eps: float, agg: str):
        self.eps = eps
        self.agg = agg

    def __call__(
        self, main_geometry: torch.Tensor, other_geometry: torch.Tensor
    ) -> torch.Tensor:
        if main_geometry.ndim == 2:
            main_geometry = main_geometry.unsqueeze(0)
        if other_geometry.ndim == 2:
            other_geometry = other_geometry.unsqueeze(0)

        main_geometry = main_geometry.unsqueeze(1)
        main_geometry = main_geometry.unsqueeze(3)
        other_geometry = other_geometry.unsqueeze(0)
        other_geometry = other_geometry.unsqueeze(2)

        dist = ((main_geometry - other_geometry) ** 2).sum(dim=-1)

        is_close = torch.sqrt(dist) < self.eps
        d1 = is_close.any(dim=3).to(torch.float).mean(dim=2)
        d2 = is_close.any(dim=2).to(torch.float).mean(dim=2)

        if self.agg == "sum":
            result = d1 + d2
        elif self.agg == "mean":
            result = (d1 + d2) / 2
        elif self.agg == "max":
            result = torch.max(d1, d2)
        else:
            raise ValueError(f"Unknown aggregation method {self.agg}")

        return result
