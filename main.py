import datetime
import os

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import time
from concept_graphs.utils import set_seed
import logging


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    log.info(f"Running config with name {cfg.name}...")
    log.info("Loading data and models...")
    set_seed(cfg.seed)
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)

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

    main_map.filter_min_segments()
    main_map.collate_objects()
    main_map.self_merge()

    stop = time.time()
    log.info("Objects in final map: %d" % len(main_map))
    log.info(f"fps: {len(dataset) / (stop - start):.2f}")

    if hasattr(cfg, "vlm") and cfg.vlm is not None:
        log.info("Captioning objects...")
        captioner = hydra.utils.instantiate(cfg.vlm)
        main_map.caption_objects(captioner)

    # main_map.draw_geometries(random_colors=False)

    # Save visualizations and map
    if not cfg.save_map:
        return

    output_dir = Path(cfg.output_dir)
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    output_dir_map = output_dir / f"{cfg.name}_{date_time}"

    log.info(f"Saving map, images and config to {output_dir_map}...")
    grid_image_path = output_dir_map / "grid_image"
    os.makedirs(grid_image_path, exist_ok=False)
    main_map.save_object_grids(grid_image_path)

    map_path = output_dir_map / "map.pkl"
    main_map.save(map_path)

    OmegaConf.save(cfg, output_dir_map / "config.yaml")

    # Create symlink to latest map
    symlink = output_dir / "latest_map.pkl"
    symlink.unlink(missing_ok=True)
    os.symlink(map_path, symlink)
    log.info(f"Created symlink to latest map at {symlink}")


if __name__ == "__main__":
    main()
