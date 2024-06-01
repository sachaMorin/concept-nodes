import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    segmentation_model = hydra.utils.instantiate(cfg.segmentation)
    obs = dataset[0]
    result = segmentation_model(obs["rgb"])

    import matplotlib.pyplot as plt
    from concept_graphs.perception.segmentation.visualization import plot_segments, plot_bbox
    plot_segments(obs["rgb"], result["masks"])
    plt.show()
    plot_bbox(obs["rgb"], result["bbox"])
    plt.show()

if __name__ == "__main__":
    main()
