import logging

import hydra
import torch
from omegaconf import DictConfig
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ovmm_ros.utils.rgbd import decode_RGBD_msg
from ovmm_ros_msg.msg import RGBDImage
from ovmm_ros_msg.srv import LocalPerception

from concept_graphs.utils import set_seed

from sensor_msgs.msg import PointCloud2

from ovmm_ros.utils.pcd import point_cloud_msg

# A logger for this file
log = logging.getLogger(__name__)


class LocalPerceptionNode(Node):

    def __init__(self, hydra_cfg):
        super().__init__('local_perception_node')

        self.local_perception_srv = self.create_service(LocalPerception, "local_perception_server", self.local_perception_srv_cb)

        self.cv_bridge = CvBridge()

        # Frame listener setup
        self.tf_frame = "locobot/arm_base_link"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Perception Pipeline
        self.hydra_cfg = hydra_cfg
        segmentation_model = hydra.utils.instantiate(self.hydra_cfg.segmentation)
        ft_extractor = hydra.utils.instantiate(self.hydra_cfg.ft_extraction)
        self.perception_pipeline = hydra.utils.instantiate(
            self.hydra_cfg.perception, segmentation_model=segmentation_model, ft_extractor=ft_extractor
        )

        # Point cloud publishers
        self.pcd_rgb_publisher = self.create_publisher(PointCloud2, 'local_perception', 10)
        timer_period = .2  # seconds
        self.pcd_rgb_timer = self.create_timer(timer_period, self.pcd_rgb_callback)

        # Pcd attributes
        self.pcd_points = None
        self.pcd_rgb = None

        self.get_logger().info("Local Perception Node Ready!")



    def local_perception_srv_cb(self, request, response):
        self.get_logger().info(f"Received Local Perception Request: {request.query}")
        try:
            frame = decode_RGBD_msg(request.rgbd, self.cv_bridge, self.tf_buffer, self.tf_frame, self.get_logger(), depth_scale=self.hydra_cfg.dataset.depth_scale)
        except Exception as e:
            self.get_logger().error(f"Failed to decode RGBD frame: {e}...")
            response.detected = False
            return response

        frame["rgb"] = frame["rgb"][:, :, [2, 1, 0]] # BGR

        segments = self.perception_pipeline(frame["rgb"], frame["depth"], frame["intrinsics"])

        if segments is None:
            response.detected = False
            self.get_logger().info("No segments detected...")
            return response

        query_ft = self.perception_pipeline.ft_extractor.encode_text([request.query])
        segments_ft = torch.from_numpy(segments["features"]).to(self.perception_pipeline.ft_extractor.device)

        sim = self.perception_pipeline.semantic_similarity(query_ft, segments_ft).cpu().numpy()

        self.get_logger().info(f"Detected {len(segments['scores'])} segments with similarities {sim}")

        max_ = np.max(sim)
        argmax = np.argmax(sim)


        if max_ > 0.63:
            # Points are in optical frame. For pcd messages, we need to transform them to self.tf_frame
            rot = frame["pose"][:3, :3]
            t = frame["pose"][:3, 3].reshape((1, 3))
            pcd_points = segments["pcd_points"][argmax]
            pcd_points = pcd_points @ rot.T + t

            self.pcd_points = pcd_points
            self.pcd_rgb = segments["pcd_rgb"][argmax] / 255
            response.detected = True
            response.object_pcd = point_cloud_msg(self.pcd_points, self.pcd_rgb, self.tf_frame)
        else:
            self.pcd_points = None
            self.pcd_rgb = None
            response.detected = False
        self.get_logger().info(f"Sending back response to relay...")
        
        return response

    def update_pcds(self):
        self.pcd_points = None
        self.pcd_rgb = None

    def pcd_rgb_callback(self):
        if self.pcd_points is None or self.pcd_rgb is None:
            return

        pcd_msg = point_cloud_msg(self.pcd_points, self.pcd_rgb, self.tf_frame)
        self.pcd_rgb_publisher.publish(pcd_msg)


@hydra.main(version_base=None, config_path="conf", config_name="ros_local_perception.yaml")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    rclpy.init()

    local_perception_node = LocalPerceptionNode(cfg)

    rclpy.spin(local_perception_node)

    local_perception_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
