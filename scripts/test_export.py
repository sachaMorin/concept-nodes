import numpy as np
import open3d as o3d
import json
from pathlib import Path

path = Path("/")

pcd = o3d.io.read_point_cloud(str(path / "point_cloud.pcd"))
clip_ft = np.load(path / "clip_features.npy")

with open(path/"segments_anno.json", "r") as f:
    segments_anno = json.load(f)

# Build a pcd with random colors
new_pcd = o3d.geometry.PointCloud()

for ann in segments_anno["segGroups"]:
    obj = pcd.select_by_index(ann["segments"])
    obj.paint_uniform_color(np.random.rand(3))
    new_pcd += obj

o3d.visualization.draw_geometries([new_pcd])