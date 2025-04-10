import os
import numpy as np
import cv2 # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from typing import Any, Tuple
from skimage.feature import peak_local_max

class TileDataset(Dataset):
    '''
    Dataset class for loading images from a list of file paths.
    '''
    def __init__(self, image_paths: list):
        """
        Args:
            image_paths (list of str): List of image file paths.
        """
        valid_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        self.image_paths = [path for path in image_paths if path.lower().endswith(valid_extensions)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = np.array(image)
        return image, image_path



class SegNCA(nn.Module):
    def __init__(
        self,
        channel_n=16,
        fire_rate=0.5,
        device="cpu",
        hidden_size=128,
        input_channels=3,
        init_method="standard",
):

        super(SegNCA, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.p0 = nn.Conv2d(
            channel_n,
            channel_n,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channel_n,
            padding_mode="reflect",
        )

        self.p1 = nn.Conv2d(
            channel_n,
            channel_n,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channel_n,
            padding_mode="reflect",
        )

        self.fc0 = nn.Linear(channel_n * 3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)

        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        z1 = self.p0(x)
        z2 = self.p1(x)
        y = torch.cat((x, z1, z2), 1)
        return y

    def update(self, x_in, fire_rate):
        x = x_in.transpose(1, 3)

        dx = self.perceive(x)
        dx = dx.transpose(1, 3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        x = x + dx.transpose(1, 3)
        x = x.transpose(1, 3)

        return x

    def make_seed(self, img):
        # creates the seed, i.e. padding the image with zeros to the desired number of channels
        seed = torch.zeros(
            (img.shape[0], img.shape[1], img.shape[2], 6), dtype=torch.float32
        )
        seed[..., 0 : img.shape[-1]] = img

        return seed

    def forward(self, x, steps=32, fire_rate=0.5):
        x = self.make_seed(x)

        for step in range(steps):
            x2 = self.update(x, fire_rate).clone()
            x = torch.concat(
                (x[..., : self.input_channels], x2[..., self.input_channels :]), 3
            )
        out = x[..., 3]

        return out, x
    


class PostProcessor:
    """
    Post-processor for NCA predictions.
    """
    def __init__(self, mode, level):
        self.mode = mode
        self.level = level

    def get_coordinates(self, x):
        input = x.squeeze().numpy()
        input[input <= 0] = 0

        if self.mode == "max":
            gauss_size = int(64 / (2**self.level) + 1)
            distance = int(40 / (2**self.level))
            input = cv2.GaussianBlur(input, (gauss_size, gauss_size), 0)
            centroids = peak_local_max(input, min_distance=distance, threshold_abs=0.5)
            centroids[:, [0, 1]] = centroids[:, [1, 0]]
        else:
            input[input > 0] = 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                input.astype(np.uint8)
            )
            min_size = 800 / (4**self.level)
            max_size = 20000 / (
                4**self.level
            )  # Minimum size of a valid cell (adjust based on your data)
            filtered_centroids = [
                centroids[i]
                for i in range(num_labels)
                if min_size < stats[i, cv2.CC_STAT_AREA] < max_size
            ]
            centroids = np.array(filtered_centroids)

        return centroids





