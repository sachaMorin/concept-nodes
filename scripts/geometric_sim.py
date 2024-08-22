import torch

from concept_graphs.mapping.similarity.geometric import point_cloud_overlap, PointCloudOverlapClosestK


# Sample 100 points on a sphere
def get_sphere(n_points, radius, translation):
    sphere = torch.randn(n_points, 3)
    sphere /= torch.norm(sphere, dim=1, keepdim=True)
    sphere *= radius
    sphere += torch.Tensor([translation])

    return sphere


s1 = get_sphere(1000, 1, [0, 0, 0])
s2 = get_sphere(1000, 1, [0, 0, 0])
s3 = get_sphere(1000, 2, [0, 0, 0])
s4 = get_sphere(1000, 1, [10.0, 0, 0])

half_sphere = s1[s1[:, 0] > 0]

while half_sphere.shape[0] < 100:
    half_sphere = torch.cat([half_sphere, half_sphere], dim=0)
half_sphere = half_sphere[:100]

# sim1, sim2 = point_cloud_overlap(s1, half_sphere, eps=0.3)
sim = PointCloudOverlapClosestK(eps=.1, agg="other", k=2, max_dist_centroid=1000)

pcds = [s1, s2, s3, s4, half_sphere]
centroids = torch.stack([p.mean(dim=0) for p in pcds])
geo_sims = sim(main_centroid=centroids, main_pcd=pcds, other_centroid=centroids, other_pcd=pcds, is_symmetrical=False)
print(geo_sims)


# sim = RadiusOverlap(eps=0.3, agg="mean")

# sim = PointCloudOverlapClosestK(eps=0.3, agg="mean")


# print(sim(s1, s1))
# print(sim(s1, s2))
# print(sim(s1, half_sphere))

# sim = RadiusOverlap(eps=0.3, agg="max")

# print(sim(s1, s1))
# print(sim(s1, s2))
# print(sim(s1, half_sphere))

# print(chamferdist(s1, s1))
# print(chamferdist(s1, s2))
# print(chamferdist(s1, s3))
# print(chamferdist(s1, s4))
#
# print(chamferdist(s1, half_sphere, agg="sum"))
# print(chamferdist(s1, half_sphere, agg="min"))

# batch = torch.stack([s1, s4], dim=0)
#
# print("Batch result")
# print(chamferdist_batch(batch, batch))
# print("Off diagonal results should equal")
# print(chamferdist(s1, s4))
#
# print("Test with half spheres (sum agg)")
# batch2 = torch.stack([s1, half_sphere, s3, s4], dim=0)
# print(chamferdist_batch(batch, batch2, agg="sum"))
# print(f"Element 0, 1 should equal")
# print(chamferdist(s1, half_sphere, agg="sum"))
#
# print("Test with half spheres (sum agg)")
# batch2 = torch.stack([s1, half_sphere, s3, s4], dim=0)
# print(chamferdist_batch(batch, batch2, agg="min"))
# print(f"Element 0, 1 should equal")
# print(chamferdist(s1, half_sphere, agg="min"))
