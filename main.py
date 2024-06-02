import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import time
import logging

from concept_graphs.mapping.ObjectMap import ObjectMap


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    log.info("Loading data and models...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)

    segmentation_model = hydra.utils.instantiate(cfg.segmentation)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)
    perception_pipeline = hydra.utils.instantiate(cfg.perception, segmentation_model=segmentation_model, ft_extractor=ft_extractor)

    log.info("Mapping...")
    main_map = hydra.utils.instantiate(cfg.mapping)
    start = time.time()
    n_segments = 0
    for obs in tqdm(dataloader):
        rgb, depth = obs["rgb"][0].numpy(), obs["depth"][0].numpy()
        intrinsics, camera_pose = obs["intrinsics"][0].numpy(), obs["camera_pose"][0].numpy()

        output = perception_pipeline(rgb, depth, intrinsics)

        local_map = ObjectMap.from_perception(output["rgb_crops"], output["mask_crops"], output["features"],
                                              output["scores"], output["pcd_points"], output["pcd_rgb"],
                                              camera_pose=camera_pose, device=cfg.mapping.device)
        main_map += local_map
        n_segments += len(local_map)
    stop = time.time()

    log.info("Mapped segments: %d" % n_segments)
    log.info("Objects in final map: %d" % len(main_map))
    log.info(f"fps: {len(dataset) / (stop - start):.2f}")

    # Viz
    main_map.draw_geometries(random_colors=True)

    # query = ["a portrait"]
    # text_feature = perception_pipeline.ft_extractor.encode_text(query).cpu().numpy()
    # similarities = output["features"] @ text_feature.T
    #
    # from concept_graphs.viz.object_pcd import visualize_object_pcd_similarities, visualize_object_pcd
    # visualize_object_pcd(output["pcd_points"])
    # visualize_object_pcd_similarities(output["pcd_points"], similarities)

if __name__ == "__main__":
    main()
