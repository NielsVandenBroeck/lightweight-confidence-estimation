import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision import transforms
from typing import List, Optional, Callable, Tuple, Any


class TorchvisionAdapter(Dataset):
    """Adapter to wrap standard torchvision datasets with custom preprocessing and filtering."""

    def __init__(self,
                 tv_dataset: Dataset,
                 img_size: Tuple[int, int],
                 classes: Optional[List[str]],
                 preprocessing_function: Optional[Callable] = None,
                 name: str = "torchvision"):
        self.name = name

        # If classes parameter is given, keep only those
        self.all_classes = getattr(tv_dataset.datasets[0], "classes", []) if isinstance(tv_dataset,
                                                                                        ConcatDataset) else getattr(
            tv_dataset, "classes", [])

        # Extract targets efficiently without loading images
        if isinstance(tv_dataset, ConcatDataset):
            all_targets = []
            for ds in tv_dataset.datasets:
                all_targets.extend(ds.targets)
        elif hasattr(tv_dataset, 'targets'):
            all_targets = tv_dataset.targets
        else:
            # Fallback (slow) if targets attribute isn't directly available
            all_targets = [label for _, label in tv_dataset]

        if classes:
            selected_classes = [c for c in classes if c in self.all_classes]
            selected_indices = []
            self.labels = []
            for i, label in enumerate(all_targets):
                if self.all_classes[label] in selected_classes:
                    selected_indices.append(i)
                    self.labels.append(selected_classes.index(self.all_classes[label]))

            self.ds = Subset(tv_dataset, selected_indices)
            self.classes = selected_classes
        else:
            self.ds = tv_dataset
            self.classes = self.all_classes
            self.labels = all_targets

        self.num_classes = len(self.classes)

        # Default transform if no preprocessing function is provided
        self.transform_function = preprocessing_function or transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img, label = self.ds[idx]

        # Map label to new class index if filtered
        if hasattr(self, "classes") and hasattr(self.ds, "dataset"):
            original_class = self.all_classes[label]
            label = self.classes.index(original_class)

        img = self.transform_function(img)

        # One-hot encode label
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1.0

        return img, label_tensor, f"{self.name}:{idx}"