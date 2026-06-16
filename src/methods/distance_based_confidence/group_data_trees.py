import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree, BallTree
from scipy.spatial.distance import pdist, cdist
from typing import List, Tuple, Dict, Any, Optional


def build_global_tree(datapoints: List[Tuple[int, np.ndarray]], leaf_size: int = 40) -> Dict[str, Any]:
    """Builds a single global KD-Tree for all training data."""
    X = np.vstack([emb for _, emb in datapoints]).astype(np.float32)
    Y = np.array([lbl for lbl, _ in datapoints])

    tree = KDTree(X, leaf_size=leaf_size)
    return {"tree": tree, "labels": Y, "X": X}


def build_class_trees(datapoints: List[Tuple[int, np.ndarray]], tree_type: str = "kd", leaf_size: int = 40) -> Dict[
    int, Dict[str, Any]]:
    """Builds individual KD or Ball trees for each class."""
    assert tree_type in ("kd", "ball")

    buckets: Dict[int, List[np.ndarray]] = {}
    for lbl, emb in datapoints:
        buckets.setdefault(lbl, []).append(emb)

    class_index = {}
    for lbl, arr in buckets.items():
        X = np.vstack(arr).astype(np.float32)
        tree = KDTree(X, leaf_size=leaf_size) if tree_type == "kd" else BallTree(X, leaf_size=leaf_size)
        class_index[lbl] = {"tree": tree, "X": X}

    return class_index


def build_class_kmeans(datapoints: List[Tuple[int, np.ndarray]], n_clusters_per_class: int = 16, minibatch: bool = True,
                       batch_size: int = 1024) -> Dict[int, Dict[str, Any]]:
    """Fits (MiniBatch)KMeans per class to reduce memory footprint."""
    buckets: Dict[int, List[np.ndarray]] = {}
    for lbl, emb in datapoints:
        buckets.setdefault(lbl, []).append(emb)

    class_kmeans = {}
    for lbl, arr in buckets.items():
        X = np.vstack(arr).astype(np.float32)
        n_clusters = min(n_clusters_per_class, X.shape[0])

        if minibatch:
            km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=min(batch_size, X.shape[0]), random_state=42)
        else:
            km = KMeans(n_clusters=n_clusters, random_state=42)

        km.fit(X)
        class_kmeans[lbl] = {"kmeans": km, "centroids": km.cluster_centers_}

    return class_kmeans


def plot_class_kmeans(datapoints: List[Tuple[int, np.ndarray]],
                      class_kmeans: Dict[int, Dict[str, Any]],
                      test_datapoints: Optional[List[Tuple[int, np.ndarray]]] = None,
                      train_ood_datapoints: Optional[List[Tuple[int, np.ndarray]]] = None,
                      ood_label: int = -1,
                      output_path: Optional[str] = None) -> None:
    """Plots datapoints and cluster centroids per class in 2D using PCA."""
    X_train = np.vstack([emb for _, emb in datapoints])
    y_train = np.array([lbl for lbl, _ in datapoints])

    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)

    plt.figure(figsize=(9, 7))
    classes = list(class_kmeans.keys())
    colors = plt.cm.get_cmap('tab10', len(classes))

    for i, lbl in enumerate(classes):
        mask = (y_train == lbl)
        plt.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], s=10, color=colors(i), alpha=0.3,
                    label=f"Class {lbl} (Train)")
        centroids = class_kmeans[lbl]["centroids"]
        centroids_2d = pca.transform(centroids)
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=150, color=colors(i), marker='X', edgecolor='k')

    if train_ood_datapoints:
        X_train_ood = np.vstack([emb for _, emb in train_ood_datapoints])
        X_train_ood_2d = pca.transform(X_train_ood)
        plt.scatter(X_train_ood_2d[:, 0], X_train_ood_2d[:, 1], s=25, color='orange', marker='s', alpha=0.6,
                    label="OOD Samples (Train)")

    if test_datapoints:
        X_test_ood = [emb for lbl, emb in test_datapoints if lbl == ood_label]
        if X_test_ood:
            X_test_ood_2d = pca.transform(np.vstack(X_test_ood))
            plt.scatter(X_test_ood_2d[:, 0], X_test_ood_2d[:, 1], s=20, color='red', marker='*', alpha=0.8,
                        label="OOD Samples (Test)")

    plt.title("KMeans Clusters vs OOD Samples (2D PCA Projection)", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("PCA Component 1", fontsize=15, fontweight='bold')
    plt.ylabel("PCA Component 2", fontsize=15, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=10, loc='best', framealpha=0.9)
    plt.tight_layout()

    if output_path:
        plt.savefig(f"{output_path}/kmeans_clusters.png", dpi=300)
    plt.show()


def plot_distance_distributions(datapoints: List[Tuple[int, np.ndarray]], ood_label: int = -1,
                                output_path: Optional[str] = None) -> None:
    """Generates a pairwise distance histogram including ID vs OOD separation."""
    id_dict: Dict[int, List[np.ndarray]] = {}
    ood_embs = []

    for lbl, emb in datapoints:
        if lbl == ood_label:
            ood_embs.append(emb)
        else:
            id_dict.setdefault(lbl, []).append(emb)

    ood_embs = np.vstack(ood_embs) if ood_embs else np.array([])
    same_class_dists, diff_class_dists = [], []
    classes = list(id_dict.keys())

    for lbl in classes:
        embs = np.vstack(id_dict[lbl])
        if len(embs) > 1:
            same_class_dists.extend(pdist(embs, metric='euclidean'))

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            embs1, embs2 = np.vstack(id_dict[classes[i]]), np.vstack(id_dict[classes[j]])
            diff_class_dists.extend(cdist(embs1, embs2, metric='euclidean').flatten())

    id_vs_ood_dists = []
    if len(ood_embs) > 0 and len(classes) > 0:
        all_id_embs = np.vstack([np.vstack(id_dict[c]) for c in classes])
        id_vs_ood_dists = cdist(all_id_embs, ood_embs, metric='euclidean').flatten()

    plt.figure(figsize=(10, 6))

    if same_class_dists:
        sns.kdeplot(same_class_dists, fill=True, color='blue', label='ID vs Same ID', alpha=0.5)
    if diff_class_dists:
        sns.kdeplot(diff_class_dists, fill=True, color='green', label='ID vs Diff ID', alpha=0.4)
    if len(id_vs_ood_dists) > 0:
        sns.kdeplot(id_vs_ood_dists, fill=True, color='red', label='ID vs OOD (Separation)', alpha=0.6, linewidth=2.5)

    plt.title("High-Dimensional Pairwise Distance Distributions", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Euclidean Distance (L2 Normalized Space)", fontsize=15, fontweight='bold')
    plt.ylabel("Density", fontsize=15, fontweight='bold')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=13, loc='upper left', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2.1)
    plt.tight_layout()

    if output_path:
        plt.savefig(f"{output_path}/distance_distributions.png", bbox_inches='tight', dpi=300)
    plt.show()