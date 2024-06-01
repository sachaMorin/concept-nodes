import torch
from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    model_type = "vit_t"
    sam_checkpoint = "/home/sacha/Documents/cg-plus/model_checkpoints/mobile_sam.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()


    obs = dataset[0]
    predictor = SamPredictor(mobile_sam)
    predictor.set_image(obs["rgb"])
    grid_batch = torch.ones(16, 1, 2).to("cuda")
    labels = torch.ones(16, 1).to("cuda")

    masks, iou_predictions, _ = predictor.predict_torch(point_coords=grid_batch, point_labels=labels)

    # def view_image_segmentation(image, mask):
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     plt.imshow(image)
    #     plt.imshow(mask * np.random.random((1, 3)), alpha=0.3)
    #     plt.show()
    #
    # view_image_segmentation(obs["rgb"], torch.permute(masks[0], (1, 2, 0)).cpu().numpy())



if __name__ == "__main__":
    main()
