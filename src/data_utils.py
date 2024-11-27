import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = [
    "airplane",
    "bus",
    "car",
    "fish",
    "guitar",
    "laptop",
    "pizza",
    "sea turtle",
    "star",
    "t-shirt",
    "The Eiffel Tower",
    "yoga",
]


class sub12Dataset(Dataset):
    """
    Each data is a tuple (image, label).
    Image is of size [1, h, w], and label is an integer from 0 to 11, each representing a class.
    """

    def _read_image(self, path):
        """
        Given an image path, returns a grey image with shape [1, h, w].
        """
        grey_image = Image.open(path).convert("L")
        grey_image = grey_image.resize((28, 28))
        grey_np = np.array(grey_image)[None, :, :] / 255.0
        # grey_np[grey_np < 0.99] = 0.0
        grey_np = 1.0 - grey_np
        grey_np = grey_np * 5.0
        grey_np[grey_np >= 0.99] = 1.0
        return grey_np.astype(np.float32)

    def __init__(self, root_dir):
        metadata = []
        from tqdm import tqdm

        for i, class_name in enumerate(CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            print(f"Loading {class_name}")
            for file_name in tqdm(os.listdir(class_dir)):
                metadata.append((os.path.join(class_dir, file_name), i))
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        path, label = self.metadata[idx]
        return (torch.from_numpy(self._read_image(path)), label)
