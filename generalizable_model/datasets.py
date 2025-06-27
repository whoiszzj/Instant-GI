import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np


class ImageAndPF(Dataset):
    def __init__(self, root_dir, scale=1.0):
        self.root_dir = root_dir
        self.transform = transforms.ToTensor()
        self.imgs = os.listdir(root_dir)
        self.scale = scale

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        img_dir = os.path.join(self.root_dir, image_name)
        image = Image.open(os.path.join(img_dir, "gt.png"))
        if self.scale != 1.0:
            image = image.resize((int(image.width * self.scale), int(image.height * self.scale)))

        image = self.transform(image)
        gt_pf_path = os.path.join(img_dir, "gt_pf.npy")
        gt_pf = torch.tensor(np.load(gt_pf_path)).float()
        return image, gt_pf


def get_data_loader(data_path, batch_size, scale=1.0):
    dataset = ImageAndPF(data_path, scale)
    max_workers = len(os.sched_getaffinity(0))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=max_workers,
        prefetch_factor=4,
        pin_memory=True
    )
