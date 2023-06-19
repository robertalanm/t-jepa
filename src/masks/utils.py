import torch

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-tokens), D (feature-dim)]
    :param masks: list of tensors containing indices of tokens in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep.long())]
    return torch.cat(all_x, dim=0)