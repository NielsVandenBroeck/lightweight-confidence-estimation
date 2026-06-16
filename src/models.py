import logging
from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

segmentation_models = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "pspnet": smp.PSPNet,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
    "linknet": smp.Linknet,
    "manet": smp.MAnet,
    "pan": smp.PAN,
    "upernet": smp.UPerNet,
    "segformer": smp.Segformer,
}


def segmentation_model_factory(
        model_name: str,
        encoder_name: str,
        device: torch.device,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        in_channels: int = 3,
        num_classes: int = 1) -> nn.Module:
    """Create a segmentation model loaded from an optional checkpoint."""
    modelfn = segmentation_models[model_name]
    model = modelfn(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
    )

    if checkpoint_path is not None:
        saved_model = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(saved_model["model_state_dict"])

    model = model.to(device)
    logging.info(f"Model name: {model_name}")
    logging.info(f"Backbone: {encoder_name}")

    return model


class EfficientNetWithEmbeddings(nn.Module):
    """Custom EfficientNet module that returns embeddings alongside logits."""

    def __init__(self, version: int, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        model_name = f"efficientnet_b{version}"
        model_fn = getattr(models, model_name)

        weights = "IMAGENET1K_V1" if pretrained else None
        self.base_model = model_fn(weights=weights)

        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor, return_embeddings: bool = False):
        embeddings = self.base_model.features(x)
        embeddings = self.base_model.avgpool(embeddings)
        embeddings = torch.flatten(embeddings, 1)

        logits = self.base_model.classifier(embeddings)

        if return_embeddings:
            return logits, embeddings
        return logits


def get_efficientnet(version: int, num_classes: int = 10, is_pretrained: bool = True,
                     dropout_rate: float = 0.2) -> nn.Module:
    """Returns an EfficientNet model from torchvision.models."""
    if not (0 <= version <= 7):
        raise ValueError("EfficientNet version must be between 0 and 7.")

    return EfficientNetWithEmbeddings(version, num_classes, is_pretrained, dropout_rate)


def test_segmentation(print_summary: bool = False):
    print("Test segmentation")
    device = "cpu"

    model = segmentation_model_factory(
        model_name="unet",
        encoder_name="resnet34",
        checkpoint_path=None,
        in_channels=3,
        pretrained=False,
        device=device
    )

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    print("Input size :", dummy_input.shape)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print("Output size:", output.shape)

    threshold = 0.5
    mask = (output > threshold).float()
    print("Mask size:", mask.shape)

    probs = torch.sigmoid(output)
    print("Probs size:", probs.shape)

    if print_summary:
        summary(model)


def test_classification(print_summary: bool = False):
    print("Test classification")
    device = "cpu"
    B, C = 7, 3
    num_classes = 10

    model = get_efficientnet(3, num_classes=num_classes, is_pretrained=True)
    dummy_input = torch.randn(B, C, 224, 224).to(device)
    print("Input size :", dummy_input.shape)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print("Output size:", output.shape)

    if print_summary:
        summary(model)

    assert output.shape == (B, num_classes)


if __name__ == "__main__":
    test_segmentation()
    test_classification()
    print("Done")