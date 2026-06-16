import logging
from pathlib import Path
from typing import Optional, Callable, Tuple, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from evaluation import compute_AUC
from src.data_processing.dataset import create_dataloaders
from src.models import get_efficientnet
from src.utils import get_num_classes_from_loader, call_confidence_fn


def save_checkpoint(path: Union[str, Path], epoch: int, classifier: nn.Module, optimizer: optim.Optimizer,
                    metric_val: float = None, confnet: Optional[nn.Module] = None) -> None:
    checkpoint = {
        'epoch': epoch,
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric_val': metric_val
    }
    if confnet is not None:
        checkpoint['confnet_state_dict'] = confnet.state_dict()

    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved at epoch {epoch} with VAL metric value {metric_val:.4f}")


def load_checkpoint(path: Union[str, Path], classifier: nn.Module, optimizer: optim.Optimizer,
                    confnet: Optional[nn.Module] = None) -> Tuple[int, float]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except ValueError:
        logging.warning(
            "Optimizer size mismatch: Starting with fresh optimizer state (expected when switching from baseline to confnet).")

    if confnet is not None and 'confnet_state_dict' in checkpoint:
        confnet.load_state_dict(checkpoint['confnet_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metric_val = checkpoint.get('metric_val', 0.0)
    return epoch, metric_val


class Trainer:
    def __init__(self,
                 classifier: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str,
                 output_path: Union[str, Path],
                 checkpoint_path: Optional[str] = None,
                 method: str = "baseline",
                 loss_fn: Optional[Callable] = None,
                 confnet: Optional[nn.Module] = None,
                 burn_in_epochs: int = 0):

        self.classifier = classifier
        self.confnet = confnet
        self.optimizer = optimizer
        self.device = device
        self.output_path = Path(output_path)
        self.method = method
        self.loss_fn = loss_fn

        self.burn_in_epochs = burn_in_epochs
        self.best_val_metric = 0.0
        self.start_epoch = 0

        if checkpoint_path:
            epoch, val_metric = load_checkpoint(checkpoint_path, self.classifier, self.optimizer, confnet)
            self.start_epoch = epoch
            if self.method == "baseline":
                self.best_val_metric = val_metric

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )

    def compute_default_loss(self, logits: torch.Tensor, labels: torch.Tensor, return_components: bool = False,
                             **kwargs) -> Any:
        is_ood = (labels[:, 0] == -1)
        targets = labels.argmax(dim=1)
        targets[is_ood] = -1

        id_mask = (targets != -1)
        n_seen = id_mask.sum()

        seen_loss = torch.tensor(0.0, device=logits.device)
        unseen_loss = torch.tensor(0.0, device=logits.device)

        if n_seen > 0:
            seen_loss = nn.CrossEntropyLoss()(logits[id_mask], targets[id_mask])

        total_loss = seen_loss

        if return_components:
            return total_loss, seen_loss, unseen_loss
        return total_loss

    def train_epoch(self, train_loader) -> float:
        self.classifier.train()
        if self.confnet is not None:
            self.confnet.train()

        total_loss = 0.0
        steps = 0
        for images, labels, paths in tqdm(train_loader, desc="Train"):
            images, labels = images.to(self.device), labels.to(self.device)

            logits, embeddings = self.classifier(images, return_embeddings=True)
            loss_fn = self.loss_fn or self.compute_default_loss
            loss = loss_fn(logits=logits, labels=labels, embeddings=embeddings, paths=paths)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(1, steps)

    def validate(self, val_loader, conf_fn: Optional[Callable] = None) -> Tuple[float, float, float, float]:
        self.classifier.eval()
        if self.confnet is not None:
            self.confnet.eval()

        combined_loss_sum, seen_loss_sum, unseen_loss_sum = 0.0, 0.0, 0.0
        total_combined, total_seen, total_unseen = 0, 0, 0
        all_conf, all_corr = [], []

        with torch.no_grad():
            for images, labels, paths in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits, embeddings = self.classifier(images, return_embeddings=True)

                is_ood = (labels[:, 0] == -1)
                targets = labels.argmax(dim=1)
                targets[is_ood] = -1

                n_combined = len(labels)
                n_seen = (targets != -1).sum().item()
                n_unseen = n_combined - n_seen

                loss_fn = self.loss_fn or self.compute_default_loss
                combined_loss, raw_seen_loss, raw_unseen_loss = loss_fn(
                    logits=logits, labels=labels, return_components=True, embeddings=embeddings, paths=paths
                )

                combined_loss_sum += (combined_loss.item() if isinstance(combined_loss, torch.Tensor) else float(
                    combined_loss)) * n_combined
                seen_loss_sum += (raw_seen_loss.item() if isinstance(raw_seen_loss, torch.Tensor) else float(
                    raw_seen_loss)) * n_seen
                unseen_loss_sum += (raw_unseen_loss.item() if isinstance(raw_unseen_loss, torch.Tensor) else float(
                    raw_unseen_loss)) * n_unseen

                total_combined += n_combined
                total_seen += n_seen
                total_unseen += n_unseen

                preds = logits.argmax(dim=1)
                corr = (preds == targets).float().cpu().numpy()

                if conf_fn:
                    conf_output = call_confidence_fn(fn=conf_fn, logits=logits, embeddings=embeddings, preds=preds)
                    conf = conf_output.cpu().numpy() if isinstance(conf_output, torch.Tensor) else np.array(conf_output)
                else:
                    conf = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()

                all_conf.extend(conf)
                all_corr.extend(corr)

        final_combined = combined_loss_sum / max(1, total_combined)
        final_seen = seen_loss_sum / max(1, total_seen)
        final_unseen = unseen_loss_sum / max(1, total_unseen)

        AUC = compute_AUC(str(self.output_path), np.array(all_conf), np.array(all_corr), "val")

        return final_combined, final_seen, final_unseen, AUC

    def fit(self, train_loader, val_loader, num_epochs: int, conf_fn: Optional[Callable] = None):
        train_losses, val_losses_combined, val_losses_seen, val_losses_unseen, auc_scores = [], [], [], [], []

        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            avg_loss = self.train_epoch(train_loader)
            train_losses.append(avg_loss)

            combined_val_loss, seen_val_loss, unseen_val_loss, auc = self.validate(val_loader, conf_fn)
            auc = float(auc)

            val_losses_combined.append(combined_val_loss)
            val_losses_seen.append(seen_val_loss)
            val_losses_unseen.append(unseen_val_loss)
            auc_scores.append(auc)

            current_lr = self.optimizer.param_groups[0]['lr']

            logging.info(
                f"Epoch [{epoch + 1:>3d}/{self.start_epoch + num_epochs:>3d}] | "
                f"LR: {current_lr:.2e} | "
                f"Loss: {combined_val_loss:.4f} (ID: {seen_val_loss:.4f}, OOD: {unseen_val_loss:.4f}) | "
                f"AUC: {auc:.4f}"
            )

            if epoch >= self.burn_in_epochs:
                if auc > self.best_val_metric:
                    self.best_val_metric = auc
                    ckpt_path = self.output_path / "checkpoint.pt"
                    try:
                        save_checkpoint(ckpt_path, epoch + 1, self.classifier, self.optimizer, self.best_val_metric,
                                        self.confnet)
                    except Exception as e:
                        logging.error(f"Failed to save checkpoint: {e}")

            if hasattr(self.loss_fn, 'step_epoch'):
                self.loss_fn.step_epoch()

            self.scheduler.step(self.best_val_metric)

        final_ckpt_path = self.output_path / "checkpoint_final.pt"
        save_checkpoint(final_ckpt_path, self.start_epoch + num_epochs, self.classifier, self.optimizer, auc,
                        self.confnet)

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.plot(val_losses_combined, label='Val Loss (Combined)', color='black', linewidth=2)
        plt.plot(val_losses_seen, label='Val Loss (Seen/Normal)', color='orange')
        plt.plot(val_losses_unseen, label='Val Loss (Unseen/Outlier)', color='green')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()

        if self.output_path:
            plt.savefig(self.output_path / "Loss.png")
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler()])
    img_size = [224, 224]

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    train_loader, val_loader, _, _ = create_dataloaders(
        root_path='../datasets/cifar-100',
        img_size=img_size,
        batch_size=16,
        shuffle=True,
        preprocessing_function=transform,
        validation_split=0.2,
        rand_seed=42
    )

    num_classes = get_num_classes_from_loader(train_loader)
    logging.info(f"Detected number of classes: {num_classes}")

    classifier = get_efficientnet(version=0, num_classes=num_classes, is_pretrained=True).to('cuda')
    optimizer = optim.Adam(list(classifier.parameters()), lr=1e-4)

    trainer = Trainer(classifier, optimizer, 'cuda', '../output')
    trainer.fit(train_loader, val_loader, 2)
    print("Training complete!")