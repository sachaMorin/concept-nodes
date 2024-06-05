from typing import List, Tuple
import numpy as np
import matplotlib.cm as cm


def similarities_to_rgb(
    similarities: np.ndarray, cmap_name: str
) -> List[Tuple[int, int, int]]:
    similarities = (similarities - similarities.min()) / (
        similarities.max() - similarities.min()
    )
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(sim.item())[:3] for sim in similarities]
    colors = [(int(255 * c[0]), int(255 * c[1]), int(255 * c[2])) for c in colors]

    return colors
