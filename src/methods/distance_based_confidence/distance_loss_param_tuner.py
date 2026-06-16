import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from torchvision import transforms

from methods.distance_based_confidence import (
    compute_datapoints,
    build_class_kmeans,
    HookedModel,
    kmeans_distance_score,
    compute_distance_loss,
    plot_class_kmeans
)
from methods.output_based_confidence import maxprob_confidence
from src.data_processing.dataset import create_dataloaders
from src.evaluation import evaluate_model
from src.models import get_efficientnet
from src.train import Trainer

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
DATASET_NAME = "Cifar-100"
METHOD = "distance_based"
CLASS_CONFIG = 2
NUM_EPOCHS = 10  # Keep this small! We are fine-tuning a pretrained model
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
OE_PROPORTION = 0.0

# Dynamic Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_PATH = PROJECT_ROOT / f"../datasets/{DATASET_NAME}"
OUTPUT_PATH = PROJECT_ROOT / f"../output/{DATASET_NAME}/config_{CLASS_CONFIG}/param_tuning"
BASELINE_CKPT_PATH = PROJECT_ROOT / f"../output/{DATASET_NAME}/config_{CLASS_CONFIG}/baseline/50epochs/checkpoint.pt"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Dataset Class Logic
if CLASS_CONFIG == 1:
    classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
    unseen_classes = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
elif CLASS_CONFIG == 2:
    classes = ['bear', 'fox']
    unseen_classes = ['squirrel']
elif CLASS_CONFIG == 3:
    classes = ['shark', 'whale']
    unseen_classes = ['dolphin']
else:
    raise ValueError(f"No Class config: {CLASS_CONFIG}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# DATALOADER SETUP (Run once globally)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_loader, val_loader, test_loader = create_dataloaders(
    root_path=str(ROOT_PATH),
    img_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=True,
    preprocessing_function=transform,
    validation_split=0.15,
    classes=classes,
    rand_seed=42,
    num_workers=4,
    unseen_eval_classes=unseen_classes,
    unseen_train_proportion=OE_PROPORTION
)

num_model_classes = len(classes)


# ==========================================
# OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial: optuna.Trial) -> float:
    # 1. Define hyperparameters to tune
    apply_l2_norm = False

    if apply_l2_norm:
        margin = trial.suggest_float("margin", 0.5, 2.0)
    else:
        margin = trial.suggest_float("margin", 5.0, 30.0)

    alpha = trial.suggest_float("alpha", 0.01, 1.5, log=True)

    # 2. Initialize Model & Load Baseline Pretrained Weights
    classifier = get_efficientnet(version=0, num_classes=num_model_classes, is_pretrained=True).to(device)

    if not BASELINE_CKPT_PATH.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at {BASELINE_CKPT_PATH}. Run baseline training first!")

    checkpoint = torch.load(BASELINE_CKPT_PATH, map_location=device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    # 3. Setup Distance Loss
    custom_loss_fn = partial(
        compute_distance_loss,
        margin=margin,
        alpha=alpha,
        apply_l2_norm=apply_l2_norm,
        ood_label=num_model_classes
    )

    # 4. Train (Fine-tune the clusters)
    trial_output_path = OUTPUT_PATH / f"trial_{trial.number}"
    trial_output_path.mkdir(exist_ok=True)

    trainer = Trainer(
        classifier=classifier,
        optimizer=optimizer,
        device=device,
        output_path=trial_output_path,
        method=METHOD,
        loss_fn=custom_loss_fn
    )

    trainer.fit(train_loader, val_loader, num_epochs=NUM_EPOCHS, conf_fn=maxprob_confidence)

    # 5. Extract Datapoints & Build KMeans
    hooked_model = HookedModel(classifier).to(device)

    all_train_datapoints = compute_datapoints(hooked_model, train_loader, device, keep_ood=False,
                                              normalize=apply_l2_norm)
    class_kmeans = build_class_kmeans(datapoints=all_train_datapoints, n_clusters_per_class=16)

    plot_class_kmeans(
        datapoints=all_train_datapoints,
        class_kmeans=class_kmeans,
        test_datapoints=None,
        ood_label=num_model_classes,
        output_path=str(trial_output_path)
    )
    centroids = np.array([class_kmeans[c]["centroids"].mean(axis=0) for c in sorted(class_kmeans.keys())])

    # 6. Evaluate Confidence
    distance_confidence = partial(kmeans_distance_score, centroids=centroids)
    distance_confidence.__name__ = "distance_confidence"

    eval_results = evaluate_model(
        classifier=classifier,
        output_path=trial_output_path,
        test_loader=test_loader,
        device=device,
        confidence_functions=[distance_confidence],
        hooked_model=hooked_model
    )

    hooked_model.remove_hook()

    # Extract Target Metric
    AUC = eval_results['distance_confidence']['AUC']
    return float(AUC)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT_PATH / "optuna_tuning.log"),
            logging.StreamHandler(),
        ]
    )

    logging.info("Starting Distance Loss Hyperparameter Tuning...")
    logging.info(f"Using pretrained baseline from: {BASELINE_CKPT_PATH}")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\n" + "=" * 50)
    print("BEST PARAMETERS FOUND:")
    print(study.best_params)
    print("BEST AUC:", study.best_value)
    print("=" * 50 + "\n")

    logging.info(f"Best params: {study.best_params} with AUC: {study.best_value}")