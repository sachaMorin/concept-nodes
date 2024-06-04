import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import time
from concept_graphs.seed import set_seed
import logging

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    log.info("Loading data and models...")
    set_seed(cfg.seed)
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)

    segmentation_model = hydra.utils.instantiate(cfg.segmentation)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)
    perception_pipeline = hydra.utils.instantiate(cfg.perception, segmentation_model=segmentation_model, ft_extractor=ft_extractor)

    log.info("Mapping...")
    progress_bar = tqdm(total=len(dataset))
    progress_bar.set_description(f"Mapping")
    start = time.time()
    n_segments = 0

    main_map = hydra.utils.instantiate(cfg.mapping)

    for obs in dataloader:
        segments = perception_pipeline(obs["rgb"], obs["depth"], obs["intrinsics"])

        local_map = hydra.utils.instantiate(cfg.mapping)
        local_map.from_perception(**segments, camera_pose=obs["camera_pose"])
        n_segments += len(local_map)

        main_map += local_map
        progress_bar.update(1)
        progress_bar.set_postfix(objects = len(main_map), segments = n_segments)

    main_map.filter_min_segments()
    main_map.self_merge()

    stop = time.time()
    log.info("Objects in final map: %d" % len(main_map))
    log.info(f"fps: {len(dataset) / (stop - start):.2f}")

    # Viz
    main_map.save_object_grids(cfg.output_dir)
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
