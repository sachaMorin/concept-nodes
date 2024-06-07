import torch


def chamferdist(x, y, agg="sum"):
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if y.ndim == 2:
        y = y.unsqueeze(0)

    dist = torch.cdist(x, y, p=2) ** 2

    x_to_y = torch.min(dist, dim=2)[0].mean(dim=1)
    y_to_x = torch.min(dist, dim=1)[0].mean(dim=1)

    if agg == "sum":
        return x_to_y + y_to_x
    elif agg == "min":
        return 2 * torch.min(x_to_y, y_to_x)  # Same scale as the sum
    else:
        raise ValueError(f"Unknown aggregation method {agg}")

def chamferdist_batch(t1: torch.Tensor, t2: torch.Tensor, agg="sum") -> torch.Tensor:
    """

    Parameters
    ----------
    x: torch.Tensor
        (B_x, N, 3) tensor
    y: torch.Tensor
        (B_y, N, 3) tensor

    Returns
    -------
    dist: torch.Tensor
        (B_x, B_y) tensor

    """
    if t1.ndim == 2:
        t1 = t1.unsqueeze(0)
    if t2.ndim == 2:
        t2 = t2.unsqueeze(1)

    t1 = t1.unsqueeze(1)
    t1 = t1.unsqueeze(3)
    t2 = t2.unsqueeze(0)
    t2 = t2.unsqueeze(2)

    dist = ((t1 - t2) ** 2).sum(dim=-1)
    d1 = dist.min(dim=3)[0].mean(dim=2)
    d2 = dist.min(dim=2)[0].mean(dim=2)

    if agg == "sum":
        return d1 + d2
    elif agg == "min":
        return 2 * torch.min(d1, d2)  # Same scale as the sum
    else:
        raise ValueError(f"Unknown aggregation method {agg}")
