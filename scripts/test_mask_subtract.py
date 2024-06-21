from concept_graphs.perception.segmentation.utils import bbox_overlap, mask_subtract_contained_torch
import torch

m1 = torch.ones((4, 4))
m2 = torch.zeros((4, 4))
m2[:2, :2] = 1
m3 = torch.zeros((4, 4))
m3[1:, 1:] = 1
m3[2:, 2:] = 0

masks = torch.stack([m1, m2, m3]).to(torch.bool)
bbox = torch.tensor([[0, 0, 4, 4], [0, 0, 2, 2], [1, 1, 4, 4]])

overlap = bbox_overlap(bbox)
expected_overlap = torch.tensor([[16., 4., 9.], [4., 4., 1.], [9., 1., 9.]]).long()
assert(torch.allclose(overlap, expected_overlap))

masks_sub = mask_subtract_contained_torch(bbox, masks)
print(masks_sub[0])
print(masks_sub[1])
print(masks_sub[2])
