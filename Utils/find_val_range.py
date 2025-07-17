import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

from CustomDataset import NyuDataset


def find_val_range(train_visualizer):
    min_val = float('inf')
    max_val = float('-inf')

    for img, depth in tqdm(train_visualizer):
        d_min = depth.min().item()
        d_max = depth.max().item()

        if d_min < min_val:
            min_val = d_min
        if d_max > max_val:
            max_val = d_max

    print(f"Depth value range: min = {min_val}, max = {max_val}")
    print(f"val_range = {max_val - min_val}")


if __name__ == '__main__':
    nyu_train_visualizer = NyuDataset(
        train_data=True,
        inverse_depth=False,
        transforms=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    )

    train_visualizer = DataLoader(
        nyu_train_visualizer,
        batch_size=1,
        num_workers=1
    )

    find_val_range(train_visualizer)