import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    perception_pipeline = hydra.utils.instantiate(cfg.perception)

    obs = dataset[0]
    output = perception_pipeline(obs["rgb"], obs["depth"], obs["intrinsics"])

    # Viz
    query = ["a door"]
    text_feature = perception_pipeline.ft_extractor.encode_text(query).cpu().numpy()
    similarities = output["features"] @ text_feature.T

    from concept_graphs.viz.object_pcd import visualize_object_pcd_similarities, visualize_object_pcd
    visualize_object_pcd(output["pcd_points"])
    visualize_object_pcd_similarities(output["pcd_points"], similarities)

if __name__ == "__main__":
    main()
