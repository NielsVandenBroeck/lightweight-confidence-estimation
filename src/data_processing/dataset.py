import logging
import random
from typing import Tuple, Optional, List

import torch
import torchvision.datasets as tv
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

from src.data_processing.folder import FolderAdapter
from src.data_processing.torch_vision import TorchvisionAdapter


class BaseAdapter(Dataset):
    """Common interface: returns (img_tensor, one_hot_label, path_like)."""

    def classes(self):
        raise NotImplementedError()

    def num_classes(self):
        return len(self.classes())


class UnifiedLabelWrapper(Dataset):
    """Pads one-hot labels to a unified length (total_classes) and applies an offset."""

    def __init__(self, dataset: Dataset, total_classes: int, offset: int = 0):
        self.dataset = dataset
        self.total_classes = total_classes
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, path = self.dataset[idx]

        if isinstance(label, torch.Tensor):
            new_label = torch.zeros(self.total_classes, dtype=label.dtype)
            orig_idx = torch.argmax(label).item() if label.numel() > 1 else int(label.item())
        else:
            new_label = torch.zeros(self.total_classes, dtype=torch.float32)
            orig_idx = label

        target_idx = min(orig_idx + self.offset, self.total_classes - 1)
        new_label[target_idx] = 1

        return img, new_label, path


class UnseenLabelWrapper(Dataset):
    """Forces the label to be a tensor of -1s of the specified target_length."""

    def __init__(self, dataset: Dataset, target_length: int):
        self.dataset = dataset
        self.target_length = target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, path = self.dataset[idx]
        dtype = label.dtype if isinstance(label, torch.Tensor) else torch.float32
        new_label = torch.full((self.target_length,), -1, dtype=dtype)
        return img, new_label, path


def create_dataloaders(root_path: str,
                       img_size: Tuple[int, int] = (224, 224),
                       batch_size: int = 32,
                       shuffle: bool = True,
                       preprocessing_function=None,
                       validation_split: float = 0.2,
                       classes: Optional[List[str]] = None,
                       rand_seed: Optional[int] = None,
                       num_workers: int = 4,
                       unseen_eval_classes: Optional[List[str]] = None,
                       unseen_train_proportion: float = 0.0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train, validation, and test dataloaders with unseen class injection."""

    unseen_eval_ds = None
    unseen_train_ds = None

    if "Animals-10" in root_path:
        all_classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'ragno', 'sheep', 'squirrel']
        selected_classes = classes if classes else [c for c in all_classes if c not in (unseen_eval_classes or [])]
        unseen_train_classes = [c for c in all_classes if
                                c not in selected_classes and c not in (unseen_eval_classes or [])]

        ds_seen = FolderAdapter(root_path, img_size, preprocessing_function, selected_classes)
        if unseen_eval_classes:
            unseen_eval_ds = FolderAdapter(root_path, img_size, preprocessing_function, unseen_eval_classes)
        if unseen_train_classes and unseen_train_proportion > 0:
            unseen_train_ds = FolderAdapter(root_path, img_size, preprocessing_function, unseen_train_classes)

    elif "OhioSmallAnimals" in root_path:
        all_classes = ['Bird-Sparrow', 'Bird-Wren', 'Bird-Yellowthroat', 'Brown Rat', 'Chipmunk',
                       'Eastern Hog-nosed snake', 'Eastern Milksnake', 'Eastern Racer Snake', 'Eastern Ribbonsnake',
                       'Frog', 'Gray Catbird', 'Gray Ratsnake', 'Indigo Bunting', 'Invertebrate', "Kirtland's Snake",
                       'Lizard', 'Meadow Jumping Mouse', 'Mink', 'Mouse-White', 'Mouse-Woodland', 'Northern Watersnake',
                       'Opossum', 'Painted Turtle', 'Rabbit', 'Raccoon', 'Red-bellied Snake', 'Shrew',
                       'Smooth Greensnake', 'Snake-Brownsnake', 'Snake-Gartersnake', 'Snake-Massasauga', 'Sora',
                       'Star-nosed mole', 'Striped Skunk', 'Vole', 'Weasel', 'Woodchuck']
        selected_classes = classes if classes else [c for c in all_classes if c not in (unseen_eval_classes or [])]
        unseen_train_classes = [c for c in all_classes if
                                c not in selected_classes and c not in (unseen_eval_classes or [])]

        ds_seen = FolderAdapter(root_path, img_size, preprocessing_function, selected_classes)
        if unseen_eval_classes:
            unseen_eval_ds = FolderAdapter(root_path, img_size, preprocessing_function, unseen_eval_classes)
        if unseen_train_classes and unseen_train_proportion > 0:
            unseen_train_ds = FolderAdapter(root_path, img_size, preprocessing_function, unseen_train_classes)

    elif "Cifar-100" in root_path:
        tv_train = tv.CIFAR100(root=root_path, train=True, download=True)
        tv_test = tv.CIFAR100(root=root_path, train=False, download=True)
        all_classes = tv_train.classes

        default_selected = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish',
                            'flatfish', 'ray', 'shark', 'trout', 'bee', 'beetle', 'butterfly',
                            'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger',
                            'wolf', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                            'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster',
                            'snail', 'spider', 'worm', 'crocodile', 'dinosaur', 'lizard',
                            'snake', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']

        selected_classes = classes if classes else [c for c in default_selected if c not in (unseen_eval_classes or [])]
        unseen_train_classes = [c for c in all_classes if
                                c not in selected_classes and c not in (unseen_eval_classes or [])]

        full_cifar = ConcatDataset([tv_train, tv_test])
        ds_seen = TorchvisionAdapter(full_cifar, img_size, selected_classes, preprocessing_function, name="Cifar-100")

        if unseen_eval_classes:
            unseen_eval_ds = TorchvisionAdapter(full_cifar, img_size, unseen_eval_classes, preprocessing_function,
                                                name="Cifar-100")
        if unseen_train_classes and unseen_train_proportion > 0:
            unseen_train_ds = TorchvisionAdapter(full_cifar, img_size, unseen_train_classes, preprocessing_function,
                                                 name="Cifar-100")
    else:
        raise ValueError(f"Unknown dataset root_path: {root_path}")

    # Process Splits and Formatting
    num_seen = len(selected_classes)
    num_eval = len(unseen_eval_classes) if unseen_eval_classes else 0
    total_classes = num_seen + num_eval

    ds_seen = UnifiedLabelWrapper(ds_seen, total_classes, offset=0)
    indices_seen = list(range(len(ds_seen)))

    if shuffle:
        random.Random(rand_seed).shuffle(indices_seen)

    train_frac = 1 - (2 * validation_split)
    n_train_seen = int(len(ds_seen) * train_frac)
    n_val_seen = int(len(ds_seen) * validation_split)

    train_idx_seen = indices_seen[:n_train_seen]
    val_idx_seen = indices_seen[n_train_seen: n_train_seen + n_val_seen]
    test_idx_seen = indices_seen[n_train_seen + n_val_seen:]

    train_dataset = Subset(ds_seen, train_idx_seen)
    val_dataset = Subset(ds_seen, val_idx_seen)
    test_dataset = Subset(ds_seen, test_idx_seen)

    len_train_unseen, len_val_unseen, len_test_unseen = 0, 0, 0
    rng = random.Random(rand_seed) if shuffle else random.Random()

    # Handle Unseen Train Injection
    if unseen_train_ds is not None and unseen_train_proportion > 0:
        indices_unseen_train = list(range(len(unseen_train_ds)))
        factor = unseen_train_proportion / (1.0 - unseen_train_proportion)
        n_train_unseen = int(len(train_idx_seen) * factor)

        if n_train_unseen > len(unseen_train_ds):
            raise ValueError(
                f"Not enough unseen background samples ({len(unseen_train_ds)}) to satisfy proportion {unseen_train_proportion}.")

        class_to_indices = {}
        for idx in indices_unseen_train:
            label_idx = unseen_train_ds.labels[idx]
            class_to_indices.setdefault(label_idx, []).append(idx)

        for label_idx in class_to_indices:
            rng.shuffle(class_to_indices[label_idx])

        train_idx_unseen = []
        active_classes = list(class_to_indices.keys())
        rng.shuffle(active_classes)
        class_pointers = {c: 0 for c in active_classes}

        while len(train_idx_unseen) < n_train_unseen and active_classes:
            for c in list(active_classes):
                if len(train_idx_unseen) >= n_train_unseen:
                    break
                if class_pointers[c] < len(class_to_indices[c]):
                    train_idx_unseen.append(class_to_indices[c][class_pointers[c]])
                    class_pointers[c] += 1
                else:
                    active_classes.remove(c)

        len_train_unseen = len(train_idx_unseen)
        wrapped_unseen_train = UnseenLabelWrapper(unseen_train_ds, target_length=total_classes)
        train_dataset = ConcatDataset([train_dataset, Subset(wrapped_unseen_train, train_idx_unseen)])

    # Handle Unseen Eval Injection
    if unseen_eval_ds is not None:
        indices_unseen_eval = list(range(len(unseen_eval_ds)))
        rng.shuffle(indices_unseen_eval)

        target_val_unseen = len(val_idx_seen)
        target_test_unseen = len(test_idx_seen)
        total_eval_needed = target_val_unseen + target_test_unseen

        if total_eval_needed > len(unseen_eval_ds):
            raise ValueError(f"Not enough unseen eval samples ({len(unseen_eval_ds)}). Need {total_eval_needed}.")

        val_idx_unseen = indices_unseen_eval[:target_val_unseen]
        test_idx_unseen = indices_unseen_eval[target_val_unseen: target_val_unseen + target_test_unseen]

        len_val_unseen = len(val_idx_unseen)
        len_test_unseen = len(test_idx_unseen)

        wrapped_val_unseen = UnseenLabelWrapper(Subset(unseen_eval_ds, val_idx_unseen), target_length=total_classes)
        wrapped_test_unseen = UnifiedLabelWrapper(Subset(unseen_eval_ds, test_idx_unseen), total_classes,
                                                  offset=num_seen)

        val_dataset = ConcatDataset([val_dataset, wrapped_val_unseen])
        test_dataset = ConcatDataset([test_dataset, wrapped_test_unseen])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    logging.info("--- Dataset Splits ---")
    logging.info(f"Total images: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")
    logging.info(
        f"Train : {len(train_loader.dataset)} (Seen: {len(train_idx_seen)}, OOD Background: {len_train_unseen})")
    logging.info(f"Val   : {len(val_loader.dataset)} (Seen: {len(val_idx_seen)}, OOD Eval: {len_val_unseen})")
    logging.info(f"Test  : {len(test_loader.dataset)} (Seen: {len(test_idx_seen)}, OOD Eval: {len_test_unseen})")
    logging.info("----------------------")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(
        root_path="../../datasets/Cifar-100",
        img_size=(224, 224),
        batch_size=32,
        classes=["beaver", "dolphin"],
        unseen_eval_classes=["otter", "seal"],
        unseen_train_proportion=0.5,
        rand_seed=42
    )

    print("Total training images:", len(train_loader.dataset))