import hydra
from omegaconf import DictConfig

from concept_graphs.perception.segmentation.utils import extract_crops


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    segmentation_model = hydra.utils.instantiate(cfg.segmentation)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction)

    obs = dataset[0]

    output = segmentation_model(obs["rgb"])
    output["image_crops"] = extract_crops(obs["rgb"], output["bbox"])
    output["features"] = ft_extractor(output["image_crops"])

    query = ["a lamp"]
    text_feature = ft_extractor.encode_text(query)
    similarities = output["features"] @ text_feature.T


    import matplotlib.pyplot as plt
    from concept_graphs.perception.segmentation.visualization import plot_segments, plot_segments_similarity, plot_bbox
    plot_segments(obs["rgb"], output["masks"])
    plt.show()
    plot_segments_similarity(obs["rgb"], output["masks"], similarities)
    plt.show()
    plot_bbox(obs["rgb"], output["bbox"])
    plt.show()

if __name__ == "__main__":
    main()
