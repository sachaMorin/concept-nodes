import open3d as o3d
import numpy as np

pcd_points = np.random.normal(size=(100, 3))
pcd_points /= np.linalg.norm(pcd_points, axis=1)[:, None]

# Create a point cloud from the points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_points)

translation = np.array([1., 1., 1.])
tf = np.eye(4)
tf[:3, 3] = translation
pcd.transform(tf)
# pcd.translate(translation)

centroid = np.mean(pcd.points, axis=0)

# Create a sphere at the centroid
centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
centroid_sphere.translate(centroid)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0, origin=[0, 0, 0]
)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd, centroid_sphere, frame])


