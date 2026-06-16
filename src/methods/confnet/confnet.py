import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional, List, Union, Tuple, Any


class ClassificationConfidenceNetwork(nn.Module):
    """Transforms the distribution of class probabilities into a scalar confidence score."""

    def __init__(self,
                 input_dim: int,
                 is_linear: bool = True,
                 activation_layer: nn.Module = nn.ReLU,
                 hidden_channels: Optional[Union[int, List[int]]] = None,
                 sort_input: bool = True):

        super().__init__()
        self.input_dim = input_dim
        self.sort_input = sort_input
        self.is_linear = is_linear

        if is_linear:
            self.fc = nn.Linear(input_dim, 1)
        else:
            if hidden_channels is None:
                hidden_channels = [input_dim]
            elif isinstance(hidden_channels, int):
                hidden_channels = [hidden_channels]

            self.fc = torchvision.ops.MLP(
                in_channels=input_dim,
                hidden_channels=hidden_channels,
                activation_layer=activation_layer,
            )
            self.final_proj = nn.Linear(hidden_channels[-1], 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sort_input:
            x, _ = torch.sort(x, dim=1, descending=True)

        if self.is_linear:
            confidence = self.fc(x)
        else:
            features = self.fc(x)
            confidence = self.final_proj(features)

        return self.sigmoid(confidence)


def confnet_score(logits: torch.Tensor,
                  embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  confnet: Optional[ClassificationConfidenceNetwork] = None,
                  use_features: bool = False,
                  device: Optional[str] = None) -> np.ndarray:
    """Wrapper to compute confidence scores using the trained ConfNet."""
    if confnet is None:
        raise ValueError("ConfNet instance must be provided.")

    confnet.eval()
    with torch.no_grad():
        if use_features and embeddings is not None:
            if isinstance(embeddings, np.ndarray):
                inputs = torch.tensor(embeddings, dtype=torch.float32).to(device)
            else:
                inputs = embeddings.clone().detach().to(device)
        else:
            logits_tensor = logits.clone().detach().to(device)
            inputs = F.softmax(logits_tensor, dim=1)

        expected = confnet.input_dim
        actual = inputs.shape[1]
        assert actual == expected, (
            f"ConfNet input_dim mismatch: model expects {expected} features "
            f"but got {actual}. Check use_features flag and embeddings."
        )
        conf_scores = confnet(inputs)

    return conf_scores.cpu().numpy().flatten()


def compute_confnet_loss(logits: torch.Tensor,
                         labels: torch.Tensor,
                         confnet: ClassificationConfidenceNetwork,
                         embeddings: Optional[torch.Tensor] = None,
                         use_features: bool = False,
                         return_components: bool = False,
                         alpha: float = 0.5,
                         epsilon: float = 1e-8,
                         **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Computes the confidence loss: J(x) = (1/N) \sum CSi * CE(x_i, y_i) - alpha * log(CSi)"""

    if use_features and embeddings is not None:
        inputs = embeddings.detach()
    else:
        inputs = F.softmax(logits.detach(), dim=1)

    conf_scores = confnet(inputs)
    confidence_scores = conf_scores.view(-1)

    if len(labels.shape) > 1:
        is_ood = (labels[:, 0] == -1)
        labels = labels.argmax(dim=1)
        labels[is_ood] = -1
    else:
        labels = labels.view(-1)

    id_mask = labels != -1
    ood_mask = labels == -1

    if not id_mask.any():
        zero_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        if return_components:
            return zero_loss, zero_loss, zero_loss
        return zero_loss

    id_logits = logits[id_mask]
    id_labels = labels[id_mask]
    id_conf_scores = confidence_scores[id_mask]

    ce_loss_per_sample = F.cross_entropy(id_logits, id_labels, reduction='none')

    weighted_ce = id_conf_scores * ce_loss_per_sample
    reg_term = -alpha * torch.log(id_conf_scores + epsilon)

    total_loss = (weighted_ce + reg_term).mean()

    if return_components:
        return total_loss, total_loss, torch.tensor(0.0, device=logits.device)

    return total_loss


def compute_confnet_bce_loss(logits: torch.Tensor,
                             labels: torch.Tensor,
                             confnet: ClassificationConfidenceNetwork,
                             embeddings: Optional[torch.Tensor] = None,
                             use_features: bool = False,
                             return_components: bool = False,
                             **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Computes BCE loss to teach the ConfNet to distrust ID mistakes."""
    if use_features and embeddings is not None:
        inputs = embeddings.detach()
    else:
        inputs = F.softmax(logits.detach(), dim=1)

    conf_scores = confnet(inputs).view(-1)
    preds = logits.argmax(dim=1)

    labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
    labels = labels.view(-1)

    id_mask = labels != -1
    ood_mask = labels == -1
    is_correct = (preds == labels)

    target_confidence = torch.zeros_like(conf_scores)
    target_confidence[is_correct] = 1.0

    bce_loss_per_sample = F.binary_cross_entropy(conf_scores, target_confidence, reduction='none')
    total_loss = bce_loss_per_sample.mean()

    if return_components:
        bce_seen_loss = bce_loss_per_sample[id_mask].mean() if id_mask.any() else torch.tensor(0.0,
                                                                                               device=logits.device)
        bce_unseen_loss = bce_loss_per_sample[ood_mask].mean() if ood_mask.any() else torch.tensor(0.0,
                                                                                                   device=logits.device)
        return total_loss, bce_seen_loss, bce_unseen_loss

    return total_loss


def test_configurations() -> None:
    batch_size = 4
    num_classes = 10
    feature_dim = 256

    logits = torch.randn(batch_size, num_classes)
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    print("--- 1. Original Method ---")
    confnet_orig = ClassificationConfidenceNetwork(input_dim=num_classes, is_linear=False, sort_input=True)
    loss_orig = compute_confnet_loss(logits, labels, confnet_orig)
    print(f"Original Loss: {loss_orig.item():.4f}")

    print("\n--- 2. No Sorting ---")
    confnet_nosort = ClassificationConfidenceNetwork(input_dim=num_classes, is_linear=False, sort_input=False)
    loss_nosort = compute_confnet_loss(logits, labels, confnet_nosort)
    print(f"No Sort Loss: {loss_nosort.item():.4f}")

    print("\n--- 3. Penultimate Layer Input ---")
    confnet_features = ClassificationConfidenceNetwork(input_dim=feature_dim, is_linear=False, sort_input=False)
    loss_features = compute_confnet_loss(logits, labels, confnet_features, embeddings=features, use_features=True)
    print(f"Penultimate Layer Loss: {loss_features.item():.4f}")

    print("\n--- 4. Lower Compute (Shrunk Method) ---")
    confnet_linear = ClassificationConfidenceNetwork(input_dim=num_classes, is_linear=True, sort_input=True)
    loss_linear = compute_confnet_loss(logits, labels, confnet_linear)
    print(f"Low Compute (Linear) Loss: {loss_linear.item():.4f}")


if __name__ == "__main__":
    test_configurations()
    print("\nDone")