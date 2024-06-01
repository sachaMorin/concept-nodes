import torch
def get_grid_coords(grid_width: int, grid_height: int, image_width: int, image_height: int, device: str) -> torch.Tensor:
    x = torch.linspace(0, 1, grid_width + 1, device=device)[1:] * image_height
    y = torch.linspace(0, 1, grid_height + 1, device=device)[1:] * image_width
    x = x.int()
    y = y.int()
    grid = torch.stack(torch.meshgrid(x, y, indexing="ij"),  dim=-1).reshape((-1, 2))
    return grid

