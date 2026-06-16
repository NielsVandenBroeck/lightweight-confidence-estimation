from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FolderAdapter(Dataset):
    def __init__(self,
                 dataset_path: str,
                 img_size: Tuple[int, int],
                 preprocessing_function: Optional[Callable],
                 classes: Optional[List[str]]):
        """Initializes the FolderAdapter Dataset class."""
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size

        # Discover classes (folder names)
        class_names = sorted([
            p.name for p in self.dataset_path.iterdir() if p.is_dir()
        ])
        self.classes = class_names if not classes else classes
        self.num_classes = len(self.classes)

        # Gather image paths and labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for cls in self.classes:
            cls_path = self.dataset_path / cls
            if not cls_path.exists():
                print(f"Folder missing for {cls}.")
                continue

            for fname in cls_path.iterdir():
                if fname.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.image_paths.append(str(fname))
                    self.labels.append(self.class_to_idx[cls])

        self.transform_function = preprocessing_function or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = self.transform_function(img)
        except Exception as e:
            # Fixed string formatting bug here
            print(f"Failed to open image {path}: {e}")
            img = torch.zeros((3, self.img_size[0], self.img_size[1]))

        # One-hot encode label
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1.0

        return img, label_tensor, path