import torch

from torch.utils.data import DataLoader


class TextDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as file:
            self.data = file.readlines()
        self.data = [line for line in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def make_text_dataset(train_path, val_path, batch_size, collator, training, drop_last=True):
    # Assuming that TextDataset is a class you have defined in src/datasets/text_dataset.py
    # and that it takes in the path to your data and a transform as arguments
    train_dataset = TextDataset(train_path, transform)
    val_dataset = TextDataset(val_path, transform)

    sampler = None
    if training:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        drop_last=drop_last,
        collate_fn=collator,
        sampler=sampler)
    
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        collate_fn=collator)
    

    return train_data_loader, val_data_loader, sampler