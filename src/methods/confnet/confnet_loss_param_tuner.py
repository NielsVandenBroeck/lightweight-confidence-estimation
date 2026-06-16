import argparse
import logging
from functools import partial
from pathlib import Path

import optuna
import torch
import torch.optim as optim
from torchvision import transforms

from methods.confnet import ClassificationConfidenceNetwork, confnet_score, compute_confnet_loss
from src.data_processing.dataset import create_dataloaders
from src.evaluation import evaluate_model
from src.models import get_efficientnet
from src.train import Trainer

DATASET_NAME = "Cifar-100"
METHOD = "confnet"
CLASS_CONFIG = 2

# Dynamic Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Mocking the args structure securely without parsing sys.argv
args = argparse.Namespace(
    root_path=str(PROJECT_ROOT / f"../datasets/{DATASET_NAME}"),
    img_size=(224, 224),
    batch_size=16,
    num_workers=4,
    learning_rate=None,  # Set dynamically in Optuna
    num_epochs=20,
    output_path=PROJECT_ROOT / f"../output/{DATASET_NAME}/{METHOD}/config_{CLASS_CONFIG}/param_tuning",
    checkpoint_path=PROJECT_ROOT / f"../output/{DATASET_NAME}/baseline/config_{CLASS_CONFIG}/50epochs/checkpoint.pt",
    device='cuda' if torch.cuda.is_available() else 'cpu',
    train=True,
    method=METHOD
)

args.output_path.mkdir(parents=True, exist_ok=True)

if CLASS_CONFIG == 1:
    classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
               'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf']
    unseen_classes = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk']
elif CLASS_CONFIG == 2:
    classes = ['beaver', 'otter']
    unseen_classes = ['camel']
elif CLASS_CONFIG == 3:
    classes = ['shark', 'whale']
    unseen_classes = ['dolphin']
else:
    raise ValueError(f"No Class config: {CLASS_CONFIG}")

transform = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor()
])

# Initialize Models and Freeze Classifier
total_classes = len(classes) + len(unseen_classes)
classifier = get_efficientnet(version=0, num_classes=total_classes, is_pretrained=True).to(args.device)

# Load checkpoint
if args.checkpoint_path.exists():
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    print("Classifier weights loaded successfully!")
else:
    print(f"WARNING: No checkpoint found at {args.checkpoint_path}! Evaluating with random weights.")

for param in classifier.parameters():
    param.requires_grad = False


def objective(trial: optuna.Trial) -> float:
    # Suggest hyperparameters
    a = trial.suggest_float("alpha", 0.05, 1.0, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    unseen_train_prop = trial.suggest_float("unseen_train_proportion", 0.0, 0.5)

    logging.info(f"Testing trial: {trial.number} with learning rate: {lr:.5f}, alpha: {a:.4f}, unseen_prop: {unseen_train_prop:.2f}")

    train_loader, val_loader, test_loader = create_dataloaders(
        root_path=args.root_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=True,
        preprocessing_function=transform,
        validation_split=0.15,
        classes=classes,
        rand_seed=42,
        num_workers=args.num_workers,
        unseen_eval_classes=unseen_classes,
        unseen_train_proportion=unseen_train_prop
    )

    # Use input_dim instead of num_classes for the updated constructor
    confnet = ClassificationConfidenceNetwork(input_dim=total_classes, is_linear=True).to(args.device)
    for param in confnet.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(confnet.parameters(), lr=lr)
    custom_loss_fn = partial(compute_confnet_loss, confnet=confnet, alpha=a)

    trainer = Trainer(
        classifier, optimizer, args.device, None, None,
        method="confnet", confnet=confnet, loss_fn=custom_loss_fn
    )

    trainer.fit(train_loader, val_loader, num_epochs=args.num_epochs)

    confnet_confidence = partial(confnet_score, confnet=confnet, device=args.device)
    confnet_confidence.__name__ = "confnet_confidence"

    # Capture the output of evaluate_model
    eval_results = evaluate_model(
        classifier,
        output_path=args.output_path,
        test_loader=test_loader,
        device=args.device,
        confidence_functions=[confnet_confidence],
        save_plots=False
    )

    # Extract ECE accurately from the returned results dictionary
    ECE = eval_results["confnet_confidence"]["ECE"]
    return float(ECE)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.output_path / "optuna_tuning.log"),
            logging.StreamHandler(),
        ]
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 50)
    print("BEST PARAMETERS FOUND:")
    print(study.best_params)
    print("BEST ECE:", study.best_value)
    print("=" * 50 + "\n")
    logging.info(f"Best params: {study.best_params} with ECE: {study.best_value}")