import torch
import torch.nn as nn
from typing import Tuple, Union, Any


def oe_loss(logits: torch.Tensor,
            labels: torch.Tensor,
            return_components: bool = False,
            **kwargs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Outlier Exposure Loss.
    If label is -1, applies OE to push to a flat distribution.
    Otherwise, applies normal cross entropy.
    """
    is_ood = (labels[:, 0] == -1)

    targets = labels.argmax(dim=1)
    targets[is_ood] = -1

    id_mask = (targets != -1)
    ood_mask = (targets == -1)

    n_seen = id_mask.sum()
    n_unseen = ood_mask.sum()
    n_total = len(labels)

    # Initialize losses
    seen_loss_sum = torch.tensor(0.0, device=logits.device)
    unseen_loss_sum = torch.tensor(0.0, device=logits.device)
    seen_loss_mean = torch.tensor(0.0, device=logits.device)
    unseen_loss_mean = torch.tensor(0.0, device=logits.device)

    if n_seen > 0:
        seen_loss_sum = nn.CrossEntropyLoss(reduction='sum')(logits[id_mask], targets[id_mask])
        seen_loss_mean = seen_loss_sum / n_seen

    if n_unseen > 0:
        ood_logits = logits[ood_mask]
        per_sample_unseen_loss = -(ood_logits.log_softmax(dim=-1).mean(dim=-1))
        unseen_loss_sum = per_sample_unseen_loss.sum()
        unseen_loss_mean = unseen_loss_sum / n_unseen

    total_loss = (seen_loss_sum + unseen_loss_sum) / n_total

    if return_components:
        return total_loss, seen_loss_mean, unseen_loss_mean

    return total_loss