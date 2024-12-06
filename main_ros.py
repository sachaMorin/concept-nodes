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

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2 as cv
from cv_bridge import CvBridge

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


from ovmm_ros_msg.msg import RGBDImage
from ovmm_ros.utils.rgbd import decode_RGBD_msg

from concept_graphs.utils import set_seed
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
        for i, n in enumerate('xyzrgb')]

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


        # Point cloud publishers
        self.pcd_rgb_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/rgb', 10)
        self.pcd_instance_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/instance', 10)
        timer_period = 1.0  # seconds
        self.pcd_instance_timer = self.create_timer(timer_period, self.pcd_instance_callback)
        self.pcd_rgb_timer = self.create_timer(timer_period, self.pcd_rgb_callback)
        self.pcds_o3d = None
        self.pcd_points = None
        self.pcd_rgb = None
        self.pcd_instance = None
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
        self.pcd_rgb = self.pcd_rgb[:, [2, 1, 0]] # BGR

        pcd_instance = []
        for i, p in enumerate(pcd_points):
            c = self.random_colors[i].reshape((1, 3))
            c = np.tile(c, (len(p), 1))
            pcd_instance.append(c)
        self.pcd_instance = np.concatenate(pcd_instance)

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
