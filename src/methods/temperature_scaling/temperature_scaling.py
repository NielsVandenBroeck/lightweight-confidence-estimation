import torch
import torch.nn as nn
from typing import Any


class TemperatureScaling(nn.Module):
    """
    Temperature scaling learns a parameter T to scale logits in case the model is over/under confident.
    T is calculated on the validation set.
    """

    def __init__(self, model: nn.Module):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Scale logits by temperature
        return logits / self.temperature.to(logits.device)

    def learn_temperature(self, val_loader: Any, device: str = 'cuda', max_iter: int = 50) -> float:
        """Collect logits & labels from validation set and optimize T."""
        self.model.eval()
        logits_list, labels_list = [], []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = self.model(images)
                logits_list.append(logits)
                labels_list.append(labels)

        logits_val = torch.cat(logits_list, dim=0)
        labels_val = torch.cat(labels_list, dim=0)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        loss_fn = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_val)
            loss = loss_fn(scaled_logits, labels_val.argmax(dim=1))
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()