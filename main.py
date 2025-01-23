import shutil
import datetime
import os
import time
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from concept_graphs.utils import set_seed
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

        if segments is None:
            continue

        local_map = hydra.utils.instantiate(cfg.mapping)
        local_map.from_perception(**segments, camera_pose=obs["camera_pose"])

        main_map += local_map

        progress_bar.update(1)
        n_segments += len(local_map)
        progress_bar.set_postfix(objects=len(main_map), map_segments=main_map.n_segments, detected_segments=n_segments)

    # Postprocessing
    main_map.filter_min_segments(n_min_segments=cfg.final_min_segments, grace=False)
    main_map.downsample_objects()
    for _ in range(2):
        main_map.denoise_objects()
        main_map.self_merge()
    main_map.downsample_objects()
    main_map.filter_min_points_pcd()

    stop = time.time()
    mapping_time = stop - start
    n_objects = len(main_map)
    fps = len(dataset) / (mapping_time)
    test_unique_segments(main_map)
    log.info("Objects in final map: %d" % n_objects)
    log.info(f"fps: {fps:.2f}")

    if cfg.caption and hasattr(cfg, "vlm_caption"):
        log.info("Captioning objects...")
        captioner = hydra.utils.instantiate(cfg.vlm_caption)
        captioner.caption_map(main_map)

    if cfg.tag and hasattr(cfg, "vlm_tag"):
        log.info("Tagging objects...")
        tagger = hydra.utils.instantiate(cfg.vlm_tag)
        tagger.caption_map(main_map)

    # Save visualizations and map
    if not cfg.save_map:
        return

    output_dir = Path(cfg.output_dir) / "concept-graphs" /  dataset.dataset_name / dataset.scene
    # Make dire
    if output_dir.exists():
        log.warning(f"Output directory {output_dir} already exists. Overwriting...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    main_map.export_openlex3d(output_dir)

    # now = datetime.datetime.now()
    # date_time = now.strftime("%Y-%m-%d-%H-%M-%S.%f")
    # output_dir_map = output_dir / f"{dataset.name}_{cfg.name}_{date_time}"

    # log.info(f"Saving map, images and config to {output_dir_map}...")
    # grid_image_path = output_dir_map / "object_viz"
    # os.makedirs(grid_image_path, exist_ok=False)
    # main_map.save_object_grids(grid_image_path)

    # # Also export some data to standard files
    # main_map.export(output_dir_map)

    # # Hydra config
    # OmegaConf.save(cfg, output_dir_map / "config.yaml")

    # # Few more stats
    # stats = dict(fps=fps, mapping_time=mapping_time, n_objects=n_objects, n_frames=len(dataset))
    # json.dump(stats, open(output_dir_map / "stats.json", "w"))

    # # Create symlink to latest map
    # symlink = output_dir / "latest_map"
    # symlink.unlink(missing_ok=True)
    # os.symlink(output_dir_map, symlink)
    # log.info(f"Created symlink to latest map at {symlink}")

    # # Move debug directory if it exists
    # if os.path.exists(output_dir / "debug"):
    #     os.rename(output_dir / "debug", output_dir_map / "debug")


if __name__ == "__main__":
    main()
