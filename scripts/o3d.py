import open3d as o3d
import numpy as np
from concept_graphs.mapping.pcd_callbacks.callbacks import DBSCAN

pcd_points = np.random.normal(size=(1000, 3))
pcd_points /= np.linalg.norm(pcd_points, axis=1)[:, None]

# Create a point cloud from the points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_points)

translation = np.array([2.0, 2.0, 2.0])
pcd.translate(translation)

pcd_points2 = np.random.normal(size=(100, 3))
pcd_points2 /= np.linalg.norm(pcd_points2, axis=1)[:, None]
pcd_points2 *= 0.5
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pcd_points2)

pcd_sum = pcd + pcd2
pcd_filtered = DBSCAN(eps=0.3, min_points=10)(pcd_sum)

# centroid = np.mean(pcd.points, axis=0)
#
# # Create a sphere at the centroid
# centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
# centroid_sphere.translate(centroid)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd_filtered])
