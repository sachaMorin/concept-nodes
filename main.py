import hydra
from omegaconf import DictConfig

from concept_graphs.perception.segmentation.utils import extract_crops
from concept_graphs.perception.rgbd_to_pcd import rgbd_to_object_pcd


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    segmentation_model = hydra.utils.instantiate(cfg.segmentation)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)

    obs = dataset[0]

    output = segmentation_model(obs["rgb"])
    output["image_crops"] = extract_crops(obs["rgb"], output["bbox"])
    output["features"] = ft_extractor(output["image_crops"])

    query = ["a door"]
    text_feature = ft_extractor.encode_text(query)
    similarities = output["features"] @ text_feature.T

    pcd_points, pcd_rgb = rgbd_to_object_pcd(obs["rgb"], obs["depth"], output["masks"].cpu().numpy(), obs["intrinsics"])

    from concept_graphs.viz.object_pcd import visualize_object_pcd_similarities
    visualize_object_pcd_similarities(pcd_points, similarities.cpu().numpy())

    import matplotlib.pyplot as plt
    from concept_graphs.viz.segmentation import plot_segments, plot_segments_similarity, plot_bbox
    plot_segments_similarity(obs["rgb"], output["masks"], similarities)
    plt.show()

if __name__ == "__main__":
    main()
