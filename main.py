import torch
from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    segmentation_model = hydra.utils.instantiate(cfg.perception.segmentation)
    obs = dataset[0]
    masks = segmentation_model(obs["rgb"])["masks"]

    import matplotlib.pyplot as plt
    from concept_graphs.perception.segmentation.visualization import plot_segments
    plot_segments(obs["rgb"], masks)
    plt.show()

if __name__ == "__main__":
    main()
