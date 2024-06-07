import torch
from concept_graphs.mapping.similarity.semantic import CosineSimilarity01Multi
from concept_graphs.perception.ft_extraction.CLIP import CLIP

sim = CosineSimilarity01Multi(agg="mean")

ft_extraction = CLIP(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    cache_dir="/home/sacha/Documents/cg-plus/model_checkpoints",
    device="cuda",
)

obj1 = ft_extraction.encode_text(["a cat", "a dog", "a bird"])
obj2 = ft_extraction.encode_text(["a banana", "an apple", "a pear"])

map1 = torch.stack([obj1, obj2], dim=0)

obj3 = ft_extraction.encode_text(["a tiger", "a wolf", "a snake"])
obj4 = ft_extraction.encode_text(["a car", "a bike", "a train"])
obj5 = ft_extraction.encode_text(["metal", "dice", "a cat"])

map2 = torch.stack([obj3, obj4, obj5], dim=0)

print(sim(map1, map1))
print(sim(map1, map2))

map1 = map1.mean(dim=1)
map1 /= map1.norm(dim=1, keepdim=True)
map2 = map2.mean(dim=1)
map2 /= map2.norm(dim=1, keepdim=True)

print(map1 @ map1.t())
print(map1 @ map2.t())

