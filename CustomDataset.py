import numpy as np
import h5py

from torch.utils.data import Dataset
from torchvision.transforms import v2

class NyuDataset(Dataset):
    def __init__(
        self,
        train_data=False,
        inverse_depth=False,
        transforms=None,
        image_shape=(480, 640),
        depth_shape=(480, 640),
        seed=42
    ):
        super().__init__()

        self.train_data = train_data
        self.transforms = transforms
        self.inverse_depth = inverse_depth
        self.image_shape = image_shape
        self.depth_shape = depth_shape

        data = h5py.File("nyu_depth_v2_labeled.mat", "r")
        self.images = data["images"]
        self.depths = data["depths"]

        indices = np.arange(len(data["images"]))  # every index

        random_generator = np.random.default_rng(seed=seed)
        random_generator.shuffle(indices)

        split_idx = int(indices.shape[0] * 0.3)

        if self.train_data:
            self.indices = indices[split_idx:]
        else:
            self.indices = indices[:split_idx]

    def __len__(self):
        if self.train_data:
            return self.indices.shape[0]
        else:
            return self.indices.shape[0]

    def __getitem__(self, idx):
        image_np = self.images[self.indices[idx]]
        depth_np = self.depths[self.indices[idx]]

        # fixing dimentions
        image_np = np.transpose(image_np, (2, 1, 0))
        image_np[2] = image_np[2][::-1]
        depth_np = np.transpose(depth_np, (1, 0))

        # calculated previously utils.find_val_range()
        self.max_val = 9.9728
        self.min_val = 0.7133

        if self.transforms:
            image, depth = self.transforms((image_np, depth_np))
            if self.train_data:
                image = v2.RandomChannelPermutation()(image)
        
        image = v2.Resize(self.image_shape)(image)
        depth = v2.Resize(self.depth_shape)(depth)

        if self.inverse_depth:
            inv_min = 1 / self.max_val
            inv_max = 1 / self.min_val
            normalized_depth = (1 / depth - inv_min) / (inv_max - inv_min)
        else:
            normalized_depth = (depth - self.min_val) / (self.max_val - self.min_val)

        return image, normalized_depth