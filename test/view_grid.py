from concept_graphs.perception.segmentation.utils import get_grid_coords
import numpy as np
import matplotlib.pyplot as plt

black_image = np.zeros((100, 120, 3), dtype=np.uint8)
grid = (
    get_grid_coords(5, 5, 120, 100, "cpu", jitter=True, uniform_jitter=True)
    .cpu()
    .numpy()
)

plt.imshow(black_image)
plt.scatter(grid[:, 0], grid[:, 1], c="red", s=10, marker="+")
plt.show()
