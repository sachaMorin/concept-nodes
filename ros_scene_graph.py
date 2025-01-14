from typing import List

import datetime
import os
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from std_srvs.srv import Empty
from sensor_msgs.msg import PointCloud2

from ovmm_ros.utils.rgbd import decode_RGBD_msg
from ovmm_ros.utils.pcd import broadcast_color_pcd, point_cloud_msg
from ovmm_ros_interfaces.msg import RGBDImage
from ovmm_ros_interfaces.srv import CLIPRetrieval, ProcessGraph

from concept_graphs.utils import set_seed, load_map
from concept_graphs.viz.utils import similarities_to_rgb


# A logger for this file
log = logging.getLogger(__name__)



class SceneGraphNode(Node):

    def __init__(self, hydra_cfg):
        super().__init__('scene_graph')
        self.hydra_cfg = hydra_cfg

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

        # Services
        self.clip_query_service = self.create_service(CLIPRetrieval, "concept_graphs/clip_query", self.clip_query_service_callback)
        self.process_graph_service = self.create_service(ProcessGraph, "concept_graphs/process_graph", self.process_graph_service_callback)
        self.export_map_service = self.create_service(Empty, "concept_graphs/export_map", self.export_map_service_callback)
        self.pickle_service = self.create_service(Empty, "concept_graphs/pickle", self.pickle_service_callback)
        self.reset_map_service = self.create_service(Empty, "concept_graphs/reset_map", self.reset_map_service_callback)

        # Where to save stuff
        self.output_dir = Path(self.hydra_cfg.output_dir)
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.output_dir_map = self.output_dir / f"ovmm_{date_time}"
        os.makedirs(self.output_dir_map, exist_ok=True)

        # Point cloud publishers
        self.pcds_o3d = None
        self.pcd_points = None
        self.pcd_rgb = None
        self.pcd_instance = None
        self.pcd_similarity = None
        self.pcd_query_points = None
        self.pcd_query_rgb = None

        self.pcd_rgb_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/rgb', 10)
        self.pcd_instance_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/instance', 10)
        self.pcd_similarity_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/similarity', 10)
        self.pcd_query_publisher = self.create_publisher(PointCloud2, 'concept_graphs/point_cloud/query', 10)

        timer_period = 1.0  # seconds
        self.pcd_instance_timer = self.create_timer(timer_period, self.pcd_instance_callback)
        self.pcd_rgb_timer = self.create_timer(timer_period, self.pcd_rgb_callback)
        self.pcd_similarity_timer = self.create_timer(timer_period, self.pcd_similarity_callback)
        self.pcd_query_timer = self.create_timer(timer_period, self.pcd_query_callback)

        # For visualization
        self.random_colors = np.random.rand(10000, 3)

        # Concept Graphs
        # Perception Pipeline
        segmentation_model = hydra.utils.instantiate(self.hydra_cfg.segmentation)
        ft_extractor = hydra.utils.instantiate(self.hydra_cfg.ft_extraction)
        self.perception_pipeline = hydra.utils.instantiate(
            self.hydra_cfg.perception, segmentation_model=segmentation_model, ft_extractor=ft_extractor
        )

        # Main scene graph
        self.pickle_path = Path(self.hydra_cfg.pickle_path) if self.hydra_cfg.pickle_path else None
        self.init_map()
        self.get_logger().info("Scene Graph ROS Node is up!")


    def init_map(self):
        if self.pickle_path is not None and self.pickle_path.exists():
            self.main_map = load_map(self.pickle_path)
            self.n_segments = self.main_map.n_segments

            self.get_logger().info(f"Loaded map from {self.pickle_path}")
        else:
            # From scratch
            self.main_map = hydra.utils.instantiate(self.hydra_cfg.mapping)
            self.n_segments = 0
        self.update_pcds()


    def listener_callback(self, msg):
        try:
            frame = decode_RGBD_msg(msg, self.cv_bridge, self.tf_buffer, self.tf_frame, self.get_logger(), depth_scale=self.hydra_cfg.dataset.depth_scale)
        except Exception as e:
            self.get_logger().error(f"Failed to decode RGBD frame: {e}...")
            return

        frame["rgb"] = frame["rgb"][:, :, [2, 1, 0]] # BGR

        segments = self.perception_pipeline(frame["rgb"], frame["depth"], frame["intrinsics"])

        if segments is None:
            return

        local_map = hydra.utils.instantiate(self.hydra_cfg.mapping)
        local_map.from_perception(**segments, camera_pose=frame["pose"])
        self.n_segments += len(local_map)

        self.main_map += local_map
        self.update_pcds()
        self.get_logger().info(f"concept graphs: objects={len(self.main_map)} map_segments={self.main_map.n_segments} detected_segments={self.n_segments}")

    def update_pcds(self):
        if not len(self.main_map):
            self.pcds_o3d = None
            self.pcd_points = None
            self.pcd_rgb = None
            self.pcd_instance = None
            self.pcd_similarity = None
            self.pcd_query_points = None
            self.pcd_query_rgb = None
            return

        self.pcds_o3d = self.main_map.pcd_o3d

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

    def process_graph_service_callback(self, request, response):
        if len(self.main_map):
            if request.self_merge:
                self.get_logger().info("Self merging...")
                self.main_map.self_merge()
            if request.n_min_segments > 0:
                self.get_logger().info("Filtering segments based on number of detections...")
                self.main_map.filter_min_segments(n_min_segments=request.n_min_segments, grace=False)

            self.update_pcds()
            self.get_logger().info(f"concept graphs: objects={len(self.main_map)} map_segments={self.main_map.n_segments} detected_segments={self.n_segments}")

        return response

    def pickle_service_callback(self, request, response):
        if len(self.main_map):
            self.get_logger().info(f"Saving map pickle to {self.output_dir_map}...")
            self.main_map.save(self.output_dir_map / "map.pkl")
        return response

    def export_map_service_callback(self, request, response):
        if len(self.main_map):
            self.get_logger().info(f"Saving map, images and config to {self.output_dir_map}...")
            grid_image_path = self.output_dir_map / "object_viz"
            os.makedirs(grid_image_path, exist_ok=False)
            self.main_map.save_object_grids(grid_image_path)

            # Also export some data to standard files
            self.main_map.export(self.output_dir_map)

            # Hydra config
            OmegaConf.save(self.hydra_cfg, self.output_dir_map / "config.yaml")

            # Create symlink to latest map
            symlink = self.output_dir / "latest_map"
            symlink.unlink(missing_ok=True)
            os.symlink(self.output_dir_map, symlink)
            self.get_logger().info(f"Created symlink to latest map at {symlink}")

            # Move debug directory if it exists
            if os.path.exists(self.output_dir / "debug"):
                os.rename(self.output_dir / "debug", self.output_dir_map / "debug")

        return response

    def reset_map_service_callback(self, request, response):
        self.init_map()
        self.get_logger().info("Scene Graph Reset Complete!")

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


@hydra.main(version_base=None, config_path="conf", config_name="ros_scene_graph.yaml")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    rclpy.init()

    scene_graph = SceneGraphNode(cfg)

    rclpy.spin(scene_graph)

    scene_graph.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
