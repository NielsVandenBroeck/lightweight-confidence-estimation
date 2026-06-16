import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from typing import Any, Tuple, Union, List, Dict, Optional
from tqdm import tqdm

# Updated import based on new folder structure
from src.methods.outlier_exposure.oe import oe_loss


def compute_distance_loss(logits: torch.Tensor,
                          labels: torch.Tensor,
                          return_components: bool = False,
                          alpha: float = 1.7,
                          margin: float = 0.5,
                          ood_label: int = -1,
                          apply_l2_norm: bool = False,
                          **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, float, float]]:
    """Computes the distance-based contrastive loss alongside standard Outlier Exposure loss."""
    embeddings = kwargs.get('embeddings', None)
    if embeddings is None:
        raise ValueError("Embeddings must be provided to compute distance loss.")

    if labels.dim() > 1:
        labels_idx = labels.argmax(dim=1)
        is_ood_mask = (labels == ood_label).all(dim=1)
        labels_idx[is_ood_mask] = ood_label
    else:
        labels_idx = labels

    main_loss = oe_loss(logits, labels)

    if apply_l2_norm:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    dists = torch.cdist(embeddings, embeddings, p=2)

    is_id = labels_idx != ood_label
    is_ood = labels_idx == ood_label
    is_both_ood = is_ood.unsqueeze(0) & is_ood.unsqueeze(1)

    mask_same_class = labels_idx.unsqueeze(0) == labels_idx.unsqueeze(1)
    both_id = is_id.unsqueeze(0) & is_id.unsqueeze(1)
    mask_same = mask_same_class & both_id

    mask_diff = (~mask_same_class & both_id) \
                | (is_id.unsqueeze(0) ^ is_id.unsqueeze(1)) \
                | is_both_ood

    triu_mask = torch.triu(torch.ones_like(mask_same), diagonal=1).bool()
    mask_same_final = mask_same & triu_mask
    mask_diff_final = mask_diff & triu_mask

    loss_same = dists[mask_same_final].sum()
    loss_diff = F.relu(margin - dists[mask_diff_final]).sum()

    num_same_pairs = mask_same_final.sum()
    num_diff_pairs = mask_diff_final.sum()
    total_pairs = num_same_pairs + num_diff_pairs

    dist_loss = (loss_same + loss_diff) / (total_pairs + 1e-8)
    total_loss = main_loss + alpha * dist_loss

    if return_components:
        return total_loss, main_loss.item(), dist_loss.item()

    return total_loss


def compute_datapoints(hooked_model: torch.nn.Module,
                       data_loader: Any,
                       device: str = "cuda",
                       ood_label: int = -1,
                       keep_ood: bool = False,
                       normalize: bool = False) -> List[Tuple[int, torch.Tensor]]:
    """Extracts embeddings for all samples in a dataloader."""
    datapoints = []
    hooked_model.eval()
    print("Calculating datapoints...")

    with torch.no_grad():
        for images, labels, _ in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            _ = hooked_model(images)
            avgpool_activations = hooked_model.get_embedding_vector().detach()

            if normalize:
                avgpool_activations = F.normalize(avgpool_activations, p=2, dim=1)

            avgpool_activations = avgpool_activations.cpu()

            decoded_labels_tensor = torch.argmax(labels, dim=1)
            is_ood_mask = (labels == -1).all(dim=1) | (labels == 0).all(dim=1)
            decoded_labels_tensor[is_ood_mask] = ood_label
            decoded_labels = decoded_labels_tensor.tolist()

            for i in range(len(decoded_labels)):
                label = decoded_labels[i]
                if label == ood_label and not keep_ood:
                    continue

                activation = avgpool_activations[i]
                datapoints.append((label, activation))

    print(f"{len(datapoints)} datapoints calculated.")
    return datapoints


def global_distance_score(embeddings: Union[np.ndarray, torch.Tensor],
                          predictions: Union[np.ndarray, torch.Tensor],
                          global_tree_dict: Dict[str, Any],
                          k_neighbors: int = 50,
                          temperature: float = 1.0) -> np.ndarray:
    """Computes confidence based on ratio of local density to top-k global neighbors."""
    embeddings = np.asarray(embeddings)
    predictions = np.asarray(predictions)

    tree = global_tree_dict["tree"]
    train_labels = global_tree_dict["labels"]

    dists, indices = tree.query(embeddings, k=k_neighbors)
    neighbor_labels = train_labels[indices]
    weights = np.exp(-dists / temperature)

    predictions_expanded = predictions[:, np.newaxis]
    match_mask = (neighbor_labels == predictions_expanded)

    numer = np.sum(weights * match_mask, axis=1)
    denom = np.sum(weights, axis=1)

    denom[denom == 0] = 1e-12
    return numer / denom


def kmeans_distance_score(embeddings: Union[np.ndarray, torch.Tensor],
                          predictions: Union[np.ndarray, torch.Tensor],
                          centroids: Union[np.ndarray, torch.Tensor],
                          temperature: float = 1.0) -> np.ndarray:
    """Computes confidence scores D(x) based on distance to class centroids."""
    embeddings = np.asarray(embeddings)
    predictions = np.asarray(predictions)
    centroids = np.asarray(centroids)

    N = embeddings.shape[0]

    dists = cdist(embeddings, centroids, metric='euclidean')
    weights = np.exp(-dists / temperature)
    numer = weights[np.arange(N), predictions]

    denom = weights.sum(axis=1)
    denom[denom == 0] = 1e-12

    return numer / denom


def tree_distance_score(embeddings: Union[np.ndarray, torch.Tensor],
                        predictions: Union[np.ndarray, torch.Tensor],
                        class_trees: Dict[int, Dict[str, Any]],
                        k_neighbors: int = 50,
                        temperature: float = 1.0) -> np.ndarray:
    """Computes confidence scores based on distance to top-k nearest neighbors in each class."""
    embeddings = np.asarray(embeddings)
    predictions = np.asarray(predictions)
    N = embeddings.shape[0]
    num_classes = len(class_trees)
    weights = np.zeros((N, num_classes))

    for c in range(num_classes):
        if c not in class_trees:
            continue

        tree = class_trees[c]["tree"]
        k = min(k_neighbors, class_trees[c]["X"].shape[0])
        dists, _ = tree.query(embeddings, k=k)

        weights[:, c] = np.mean(np.exp(-dists / temperature), axis=1)

    numer = weights[np.arange(N), predictions]
    denom = weights.sum(axis=1)
    denom[denom == 0] = 1e-12

    return numer / denom