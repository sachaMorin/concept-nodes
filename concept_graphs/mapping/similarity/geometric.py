import torch
from .Similarity import GeometricSimilarity
from .chamferdist import chamferdist_batch


class CentroidDist(GeometricSimilarity):
    def __call__(self, main_centroid: torch.Tensor, other_centroid: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(main_centroid - other_centroid, dim=1)
        return 1 / (dist + 1e-6)


class ChamferDist(GeometricSimilarity):
    def __init__(self, agg: str):
        self.agg = agg

    def __call__(self, main_geometry: torch.Tensor, other_geometry: torch.Tensor) -> torch.Tensor:
        dist = chamferdist_batch(main_geometry, other_geometry, agg=self.agg)
        return 1 / (dist + 1e-6)
