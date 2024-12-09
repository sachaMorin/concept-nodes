from typing import List

import datetime
import os
import time
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2 as cv
from cv_bridge import CvBridge

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


from ovmm_ros.utils.rgbd import decode_RGBD_msg
from ovmm_ros_msg.msg import RGBDImage
from ovmm_ros_msg.srv import CLIPRetrieval

from concept_graphs.utils import set_seed
from concept_graphs.viz.utils import similarities_to_rgb
from concept_graphs.mapping.utils import test_unique_segments

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField

# A logger for this file
log = logging.getLogger(__name__)

def point_cloud_msg(points, colors, parent_frame):

    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.

    data = np.hstack([points, colors]).astype(dtype).tobytes()
    # data = points.astype(dtype).tobytes() 

    # The fields specify what the bytes represents. The first 4 bytes 
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzbgr')]

    # The PointCloud2 message also has a header which specifies which 
    # coordinate frame it is represented in. 
    header = Header(frame_id=parent_frame)

    return PointCloud2(
        header=header,
        height=1, 
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 6), # Every point consists of three float32s.
        row_step=(itemsize * 6 * points.shape[0]), 
        data=data
    )

def broadcast_color_pcd(pcds_o3d: List[o3d.geometry.PointCloud], colors: np.array):
    """Take a list with n_objects pcds, array of size (n_objects, 3) and returns an array of size (n_points, 3)."""
    pcd_points = [np.asarray(p.points) for p in pcds_o3d]
    pcd_colors = []
    for i, p in enumerate(pcd_points):
        c = colors[i].reshape((1, 3))
        c = np.tile(c, (len(p), 1))
        pcd_colors.append(c)
    pcd_colors = np.concatenate(pcd_colors)

    return pcd_colors


class SceneGraphNode(Node):

    def __init__(self, hydra_cfg):
        super().__init__('scene_graph')

        # ROS
        self.subscription = self.create_subscription(
            RGBDImage,
            '/rgbd_scene_graph',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.cv_bridge = CvBridge()

        # Frame listener setup
        self.tf_frame = "map"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Concept Graphs
        # Perception Pipeline
        self.hydra_cfg = hydra_cfg
        segmentation_model = hydra.utils.instantiate(self.hydra_cfg.segmentation)
        ft_extractor = hydra.utils.instantiate(self.hydra_cfg.ft_extraction)
        self.perception_pipeline = hydra.utils.instantiate(
            self.hydra_cfg.perception, segmentation_model=segmentation_model, ft_extractor=ft_extractor
        )

        # Main scene graph
        self.main_map = hydra.utils.instantiate(self.hydra_cfg.mapping)
        self.n_segments = 0
        self.get_logger().info("Scene Graph ROS Node is up!")

        # Query service
        self.clip_query_service = self.create_service(CLIPRetrieval, "concept_graphs/clip_query", self.clip_query_service_callback)

        # Filter service
        

        # Point cloud publishers
        self.pcd_rgb_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/rgb', 10)
        self.pcd_instance_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/instance', 10)
        self.pcd_similarity_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/similarity', 10)
        self.pcd_query_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/query', 10)

        timer_period = 1.0  # seconds
        self.pcd_instance_timer = self.create_timer(timer_period, self.pcd_instance_callback)
        self.pcd_rgb_timer = self.create_timer(timer_period, self.pcd_rgb_callback)
        self.pcd_similarity_timer = self.create_timer(timer_period, self.pcd_similarity_callback)
        self.pcd_query_timer = self.create_timer(timer_period, self.pcd_query_callback)

        self.pcds_o3d = None
        self.pcd_points = None
        self.pcd_rgb = None
        self.pcd_instance = None
        self.pcd_similarity = None
        self.pcd_query_points = None
        self.pcd_query_rgb = None
        self.random_colors = np.random.rand(10000, 3)


    def listener_callback(self, msg):
        frame = decode_RGBD_msg(msg, self.cv_bridge, self.tf_buffer, self.tf_frame, self.get_logger(), depth_scale=self.hydra_cfg.dataset.depth_scale)

        segments = self.perception_pipeline(frame["rgb"], frame["depth"], frame["intrinsics"])

        if segments is None:
            return

        local_map = hydra.utils.instantiate(self.hydra_cfg.mapping)
        local_map.from_perception(**segments, camera_pose=frame["pose"])
        self.n_segments += len(local_map)

        self.main_map += local_map
        self.pcds_o3d = self.main_map.pcd_o3d
        self.update_pcds()
        self.get_logger().info(f"concept graphs: objects={len(self.main_map)} map_segments={self.main_map.n_segments} detected_segments={self.n_segments}")

    def update_pcds(self):
        if self.pcds_o3d is None:
            return

        pcd_points = [np.asarray(p.points) for p in self.pcds_o3d]
        self.pcd_points = np.concatenate(pcd_points)

        pcd_rgb = [np.asarray(p.colors) for p in self.pcds_o3d]
        self.pcd_rgb = np.concatenate(pcd_rgb)

        self.pcd_instance = broadcast_color_pcd(self.pcds_o3d, self.random_colors)


    def clip_query_service_callback(self, request, response):
        if not len(self.main_map):
            response.success = False
            return response

        # Query CLIP similarities
        query_ft = self.perception_pipeline.ft_extractor.encode_text([request.query])
        sim_objects = self.main_map.similarity.semantic_similarity(query_ft, self.main_map.semantic_tensor)
        sim_objects = sim_objects.squeeze().cpu().numpy().astype(float)

        best_match_idx = sim_objects.argmax()

        sim_rgb = np.array(similarities_to_rgb(sim_objects, "viridis")) / 255

        self.pcd_similarity = broadcast_color_pcd(self.pcds_o3d, sim_rgb)
        self.pcd_similarity = self.pcd_similarity[:, [2, 1, 0]]

        # Get most similar object
        self.pcd_query_points = np.array(self.pcds_o3d[best_match_idx].points)
        self.pcd_query_rgb = np.array(self.pcds_o3d[best_match_idx].colors)
        translation = self.pcd_query_points.mean(axis=0).astype(float)

        response.success = True
        response.frame_id = self.tf_frame
        response.x = translation[0] 
        response.y = translation[1]
        response.z = translation[2]
        response.similarity = sim_objects[best_match_idx]

        return response

    def pcd_instance_callback(self):
        if self.pcds_o3d is None:
            return

        pcd_msg = point_cloud_msg(self.pcd_points, self.pcd_instance, self.tf_frame)
        self.pcd_instance_publisher.publish(pcd_msg)

    def pcd_rgb_callback(self):
        if self.pcds_o3d is None:
            return

        pcd_msg = point_cloud_msg(self.pcd_points, self.pcd_rgb, self.tf_frame)
        self.pcd_rgb_publisher.publish(pcd_msg)

    def pcd_similarity_callback(self):
        if self.pcd_similarity is None or len(self.pcd_similarity) != len(self.pcd_points):
            return

        pcd_msg = point_cloud_msg(self.pcd_points, self.pcd_similarity, self.tf_frame)
        self.pcd_similarity_publisher.publish(pcd_msg)

    def pcd_query_callback(self):
        if self.pcd_query_points is None or self.pcd_query_rgb is None:
            return

        pcd_msg = point_cloud_msg(self.pcd_query_points, self.pcd_query_rgb, self.tf_frame)
        self.pcd_query_publisher.publish(pcd_msg)


@hydra.main(version_base=None, config_path="conf", config_name="main_ros")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    rclpy.init()

    scene_graph = SceneGraphNode(cfg)

    rclpy.spin(scene_graph)

    scene_graph.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
