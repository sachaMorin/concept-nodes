import datetime
import os

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import time
from concept_graphs.utils import set_seed
import logging
from concept_graphs.mapping.utils import test_unique_segments


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    log.info(f"Running algo {cfg.name}...")
    log.info("Loading data and models...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)
    log.info(f"Loaded dataset {dataset.name}.")

    segmentation_model = hydra.utils.instantiate(cfg.segmentation)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)
    perception_pipeline = hydra.utils.instantiate(
        cfg.perception, segmentation_model=segmentation_model, ft_extractor=ft_extractor
    )

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

        main_map += local_map

        progress_bar.update(1)
        n_segments += len(local_map)
        progress_bar.set_postfix(objects=len(main_map), segments=n_segments)

    # Postprocessing
    main_map.filter_min_segments(n_min_segments=cfg.final_min_segments, grace=False)
    main_map.downsample_objects()
    for _ in range(2):
        main_map.denoise_objects()
        for obj in main_map:
            obj.cluster_top_k(6)
        main_map.collate_objects()
        main_map.collate()
        main_map.self_merge()
    main_map.downsample_objects()
    main_map.filter_min_points_pcd()

    stop = time.time()
    test_unique_segments(main_map)
    log.info("Objects in final map: %d" % len(main_map))
    log.info(f"fps: {len(dataset) / (stop - start):.2f}")

    if cfg.caption and hasattr(cfg, "vlm_caption"):
        log.info("Captioning objects...")
        captioner = hydra.utils.instantiate(cfg.vlm_caption)
        captioner.caption_map(main_map)

    if cfg.tag and hasattr(cfg, "vlm_tag"):
        log.info("Tagging objects...")
        captioner = hydra.utils.instantiate(cfg.vlm_tag)
        captioner.caption_map(main_map)

    # Save visualizations and map
    if not cfg.save_map:
        return

    output_dir = Path(cfg.output_dir)
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    output_dir_map = output_dir / f"{dataset.name}_{cfg.name}_{date_time}"

    log.info(f"Saving map, images and config to {output_dir_map}...")
    grid_image_path = output_dir_map / "grid_image"
    os.makedirs(grid_image_path, exist_ok=False)
    main_map.save_object_grids(grid_image_path)

    # Also export some data to standard files for consumption by other packages
    main_map.export(output_dir_map)

    # Hydra config
    OmegaConf.save(cfg, output_dir_map / "config.yaml")

    # Create symlink to latest map
    symlink = output_dir / "latest_map"
    symlink.unlink(missing_ok=True)
    os.symlink(output_dir_map, symlink)
    log.info(f"Created symlink to latest map at {symlink}")

    # Move debug directory if it exists
    if os.path.exists(output_dir / "debug"):
        os.rename(output_dir / "debug", output_dir_map / "debug")


if __name__ == "__main__":
    main()
