import torch
from .Similarity import SemanticSimilarity


class CosineSimilarity(SemanticSimilarity):
    def __call__(
        self, main_semantic: torch.Tensor, other_semantic: torch.Tensor
    ) -> torch.Tensor:
        return main_semantic @ other_semantic.t()


class CosineSimilarity01(SemanticSimilarity):
    def __call__(
        self, main_semantic: torch.Tensor, other_semantic: torch.Tensor
    ) -> torch.Tensor:
        return main_semantic @ other_semantic.t() / 2 + 0.5


class CosineSimilarity01Multi(SemanticSimilarity):
    def __init__(self, agg: str):
        self.agg = agg

    def __call__(
        self, main_semantic: torch.Tensor, other_semantic: torch.Tensor
    ) -> torch.Tensor:
        if main_semantic.dim() == 2:
            main_semantic = main_semantic.unsqueeze(0)
        if other_semantic.dim() == 2:
            other_semantic = other_semantic.unsqueeze(0)

        sim = torch.einsum("ijk,lmk->iljm", main_semantic, other_semantic) / 2 + 0.5
        if self.agg == "mean":
            return sim.mean(dim=(2, 3))
        if self.agg == "max":
            return sim.max(dim=-1)[0].max(dim=-1)[0]
        else:
            raise ValueError(f"Unknown aggregation method {self.agg}")
