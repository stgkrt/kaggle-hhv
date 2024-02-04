import torch


def clip_underover_value(x: torch.Tensor, percent=0.1) -> torch.Tensor:
    sorted_x, _ = torch.sort(x.reshape(-1))
    under_threshold = sorted_x[int(len(x.reshape(-1)) * percent)]
    over_threshold = sorted_x[-int(len(x.reshape(-1)) * percent)]
    clip_x = torch.clamp(x, under_threshold, over_threshold)
    return clip_x
