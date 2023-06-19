import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np

class TextMemmapDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index:index+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[index+1:index+1+self.block_size].astype(np.int64))
        return x, y

class TextMaskCollator:
    def __init__(self, mask_ratio, device_type='cuda'):
        self.mask_ratio = mask_ratio
        self.device_type = device_type
        self.device = torch.device(device_type)

    def __call__(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.stack(y)
        mask = torch.full_like(x, False, dtype=torch.bool)
        num_masks = int(self.mask_ratio * x.numel())
        mask.view(-1)[torch.randint(x.numel(), (num_masks,))] = True
        x[mask] = 0
        masks_enc = mask.clone()  # copying the mask
        masks_pred = torch.zeros_like(mask)  # placeholder tensor, if not used
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
            masks_enc, masks_pred = masks_enc.pin_memory().to(self.device, non_blocking=True), masks_pred.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            masks_enc, masks_pred = masks_enc.to(self.device), masks_pred.to(self.device)
        return x, masks_enc, masks_pred
def make_text_dataloader(data_path, block_size, batch_size, mask_ratio, device_type='cuda', num_workers=0):
    dataset = TextMemmapDataset(data_path, block_size)
    collator = TextMaskCollator(mask_ratio, device_type)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator
    )
    return data_loader

# usage