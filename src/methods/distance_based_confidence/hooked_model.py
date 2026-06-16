import torch
import torch.nn as nn
from typing import Any, Tuple

from src.models import get_efficientnet


class HookedModel(nn.Module):
    """Wraps the model inside a class where we can extract the penultimate layer on training samples."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.hook = self.model.base_model.avgpool.register_forward_hook(self.hook_fn)
        self.embedding_vector = []

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model(x, *args, **kwargs)

    def hook_fn(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        """Called during forward. Appends the output of an internal module layer to the embedding_vector list."""
        self.embedding_vector = [output.view(output.size(0), -1)]

    def get_embedding_vector(self) -> torch.Tensor:
        """Returns the embedding vector (penultimate layer) of the model."""
        if not self.embedding_vector:
            raise RuntimeError("Embedding vector is empty. Ensure a forward pass has occurred.")
        return self.embedding_vector[0]

    def remove_hook(self) -> None:
        """Removes the hook from the model to prevent memory leaks."""
        self.hook.remove()


if __name__ == "__main__":
    num_classes = 16

    # create the efficientnet model
    model = get_efficientnet(version=0, num_classes=num_classes, is_pretrained=True)

    # modify the model to output the embedding vector
    new_model = HookedModel(model)

    # test the new model
    dummy_input = torch.randn(1, 3, 640, 640)
    _ = new_model(dummy_input)

    print(">>>", new_model.get_embedding_vector().shape)  # torch.Size([1, 1280]) for EfficientNet-B0

    new_model.remove_hook()