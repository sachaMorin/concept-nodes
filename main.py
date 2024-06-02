import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import time
import logging


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
    start = time.time()
    for obs in tqdm(dataloader):
        rgb, depth, intrinsics = obs["rgb"][0].numpy(), obs["depth"][0].numpy(), obs["intrinsics"][0].numpy()
        output = perception_pipeline(rgb, depth, intrinsics)
    stop = time.time()

    # Log fps
    log.info(f"FPS: {len(dataset) / (stop - start):.2f}")

    # Viz
    query = ["a portrait"]
    text_feature = perception_pipeline.ft_extractor.encode_text(query).cpu().numpy()
    similarities = output["features"] @ text_feature.T

    from concept_graphs.viz.object_pcd import visualize_object_pcd_similarities, visualize_object_pcd
    visualize_object_pcd(output["pcd_points"])
    visualize_object_pcd_similarities(output["pcd_points"], similarities)

if __name__ == "__main__":
    main()
