import logging
import re
from functools import partial
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from methods.confnet import ClassificationConfidenceNetwork, confnet_score
from methods.distance_based_confidence import (
    compute_datapoints, build_global_tree, build_class_kmeans, build_class_trees,
    HookedModel, global_distance_score, kmeans_distance_score, tree_distance_score
)
from methods.evidential_deep_learning import edl_confidence
from methods.monte_carlo_dropout import MCDropoutWrapper, mc_dropout_confidence
from methods.output_based_confidence import maxprob_confidence, wdf_confidence
from methods.temperature_scaling import TemperatureScaling
from src.data_processing.dataset import create_dataloaders
from src.evaluation import evaluate_model
from src.models import get_efficientnet


def parse_modification(folder_name: str) -> str:
    """Translates folder names into clean table names."""
    mod = folder_name.replace("50epochs", "").replace("20epochs", "").strip("_")

    if not mod or mod == "(train)":
        return "Standard"
    if "unseen" in mod:
        return mod.replace("unseen", "Outlier Exposure ")
    if "drop" in mod:
        return mod.replace("drop", "Dropout ")

    return mod.replace("_", " ")


def get_dataset_classes(dataset_name: str, config_num: int) -> Tuple[List[str], List[str]]:
    """Returns the correct seen and unseen classes based on the dataset and config."""
    if dataset_name == "Cifar-100":
        if config_num == 1:
            return ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark',
                    'trout'], \
                ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
        elif config_num == 2:
            return ['bear', 'fox'], ['squirrel']
        elif config_num == 3:
            return ['shark', 'whale'], ['dolphin']

    elif dataset_name == "OhioSmallAnimals":
        if config_num == 1:
            return ['Bird-Sparrow', 'Bird-Wren', 'Snake-Brownsnake', 'Snake-Gartersnake', 'Mouse-White', 'Shrew',
                    'Mink', 'Chipmunk', 'Frog', 'Lizard'], \
                ['Bird-Yellowthroat', 'Snake-Massasauga', 'Mouse-Woodland', 'Vole', 'Weasel']
        elif config_num == 2:
            return ['Chipmunk', 'Vole'], ['Mouse-Woodland']
        elif config_num == 3:
            return ['Snake-Gartersnake', 'Snake-Massasauga'], ['Snake-Brownsnake']

    raise ValueError(f"Unknown dataset {dataset_name} or config {config_num}")


def run_master_sweep():
    # ==========================================
    # 1. GLOBAL CONFIGURATION
    # ==========================================
    DATASET_NAME = "Cifar-100"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    BASE_DIR = Path(f"../output/{DATASET_NAME}")
    OUTPUT_CSV = BASE_DIR / "thesis_results_master_NEW.csv"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==========================================
    # 2. DISCOVER ALL CHECKPOINTS
    # ==========================================
    all_results = []

    # Pathlib rglob cleanly finds all checkpoint.pt files up to 3 folders deep
    checkpoint_files = list(BASE_DIR.glob("config_*/*/*/checkpoint.pt"))

    if not checkpoint_files:
        logging.warning(f"No checkpoints found in {BASE_DIR}")
        return

    logging.info(f"Found {len(checkpoint_files)} total checkpoints to evaluate across all configurations!")

    for ckpt_path in checkpoint_files:
        folder_path = ckpt_path.parent

        # pathlib .parts allows safe indexing: base/config_X/method/variant
        config_folder = folder_path.parts[-3]
        method_name = folder_path.parts[-2]
        variant_folder = folder_path.parts[-1]
        config_num = int(config_folder.split("_")[1])

        if method_name == "baseline" and config_num in [2, 3]:
            if "unseen" in variant_folder and "unseen0.50" not in variant_folder:
                logging.info(f"⏭️ Skipping redundant Baseline OE: {variant_folder}")
                continue

        if method_name == "distance_based":
            if "norm" not in variant_folder.lower() and "unseen" in variant_folder.lower():
                logging.info(f"⏭️ Skipping unnormalized OE Distance-Based: {variant_folder}")
                continue

        modification_name = parse_modification(variant_folder)
        is_tempscaling = False

        if method_name == "baseline" and "tempscaling" in variant_folder:
            is_tempscaling = True
            base_variant = variant_folder.replace("_tempscaling", "")
            ckpt_path = folder_path.parent / base_variant / "checkpoint.pt"

            if not ckpt_path.exists():
                logging.warning(f"Skipping {variant_folder} (No base checkpoint found)")
                continue

        logging.info(f"\n{'=' * 50}")
        logging.info(f"Evaluating -> Config: {config_num} | Method: {method_name} | Mod: {modification_name}")
        logging.info(f"{'=' * 50}")

        try:
            classes, unseen_classes = get_dataset_classes(DATASET_NAME, config_num)
            num_model_classes = len(classes)

            train_loader, val_loader, test_loader = create_dataloaders(
                root_path=f'../datasets/{DATASET_NAME}',
                img_size=[224, 224],
                batch_size=32,
                shuffle=True,
                preprocessing_function=transform,
                validation_split=0.15,
                classes=classes,
                rand_seed=42,
                num_workers=4,
                unseen_eval_classes=unseen_classes,
                unseen_train_proportion=0.0,
            )

            current_dropout = 0.2
            if "drop" in variant_folder:
                try:
                    current_dropout = float(variant_folder.split("drop")[-1])
                except ValueError:
                    current_dropout = 0.2

            classifier = get_efficientnet(
                version=0,
                num_classes=num_model_classes,
                is_pretrained=False,
                dropout_rate=current_dropout
            ).to(DEVICE)

            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            classifier.eval()

            eval_confidence_functions = []
            eval_hooked_model = None
            temperature = 1.0

            if method_name == "baseline":
                eval_confidence_functions = [maxprob_confidence, wdf_confidence]

            elif method_name == "crl":
                eval_confidence_functions = [maxprob_confidence]

            elif method_name == "confnet":
                if "embeddings" in variant_folder:
                    input_dim = 1280
                    sort_input = False
                    hidden_channels = [512, 128, 64] if config_num == 1 else [1280, 64]
                    match = re.search(r'\[(.*?)\]', variant_folder)
                    if match:
                        try:
                            nums = match.group(1).split(',')
                            hidden_channels = [int(n.strip()) for n in nums if n.strip()]
                        except Exception:
                            pass
                    is_linear = (len(hidden_channels) == 0)
                else:
                    input_dim = num_model_classes
                    sort_input = True
                    hidden_channels = []
                    is_linear = True

                confnet = ClassificationConfidenceNetwork(
                    input_dim=input_dim,
                    is_linear=is_linear,
                    hidden_channels=hidden_channels,
                    sort_input=sort_input
                ).to(DEVICE)

                confnet.load_state_dict(checkpoint['confnet_state_dict'], strict=True)
                confnet.eval()

                use_features = "embeddings" in variant_folder
                conf_fn = partial(confnet_score, confnet=confnet, device=DEVICE, use_features=use_features)
                conf_fn.__name__ = "confnet_confidence"
                eval_confidence_functions = [conf_fn]

            elif method_name == "mc_dropout":
                is_ohio_config2 = (DATASET_NAME == "Cifar-100" and config_num == 2)
                is_no_oe = ("unseen" not in variant_folder.lower())
                is_drop05 = (current_dropout == 0.5)

                passes_to_test = [1, 5, 10, 20, 50] if (is_ohio_config2 and is_no_oe and is_drop05) else [10]
                base_mod_name = modification_name.replace("(train)", "").strip() or "Standard"

                for p in passes_to_test:
                    logging.info(f"--- Evaluating MC Dropout with {p} passes ---")
                    eval_hooked_model = MCDropoutWrapper(classifier, num_passes=p)

                    fn = partial(mc_dropout_confidence)
                    fn.__name__ = f"Entropy ({p} passes)" if p != 10 else "Entropy"

                    metrics_dict = evaluate_model(
                        classifier=classifier,
                        output_path=folder_path,
                        test_loader=test_loader,
                        device=DEVICE,
                        confidence_functions=[fn],
                        hooked_model=eval_hooked_model,
                        temperature=temperature,
                        save_plots=False
                    )

                    if metrics_dict:
                        for conf_method, metrics in metrics_dict.items():
                            all_results.append({
                                "Configuration": config_folder,
                                "Method": method_name.capitalize(),
                                "Modification": base_mod_name,
                                "Conf_Fn": conf_method,
                                **metrics
                            })
                continue

            elif method_name == "distance_based":
                eval_hooked_model = HookedModel(classifier).to(DEVICE)
                is_norm = "norm" in variant_folder.lower()

                logging.info(f"Extracting datapoints (L2 Norm: {is_norm})...")
                all_train_datapoints = compute_datapoints(
                    eval_hooked_model, train_loader, DEVICE,
                    keep_ood=True, normalize=is_norm, ood_label=num_model_classes
                )

                train_id_datapoints = [(lbl, emb) for lbl, emb in all_train_datapoints if lbl != num_model_classes]
                default_temp = 0.1 if config_num == 1 else 0.4

                if DATASET_NAME == "Cifar-100" and config_num == 2:
                    logging.info("Executing Grid Search for Cifar-100 Config 2...")
                    dist_types = ["tree", "kmeans"] if "unseen" in variant_folder.lower() else ["tree", "kmeans",
                                                                                                "global"]
                    temps = [1.0, default_temp]

                    if "tree" in dist_types:
                        class_trees = build_class_trees(datapoints=train_id_datapoints, tree_type="ball", leaf_size=40)
                    if "kmeans" in dist_types:
                        class_kmeans = build_class_kmeans(datapoints=train_id_datapoints, n_clusters_per_class=16)
                        centroids = np.array(
                            [class_kmeans[c]["centroids"].mean(axis=0) for c in sorted(class_kmeans.keys())])
                    if "global" in dist_types:
                        global_tree = build_global_tree(datapoints=train_id_datapoints, leaf_size=40)

                    for dtype in dist_types:
                        for t in temps:
                            if dtype == "tree":
                                fn = partial(tree_distance_score, class_trees=class_trees, k_neighbors=50,
                                             temperature=t)
                            elif dtype == "kmeans":
                                fn = partial(kmeans_distance_score, centroids=centroids, temperature=t)
                            elif dtype == "global":
                                fn = partial(global_distance_score, global_tree_dict=global_tree, k_neighbors=50,
                                             temperature=t)

                            fn.__name__ = f"{dtype.capitalize()}||{t}"
                            eval_confidence_functions.append(fn)

                else:
                    logging.info(f"Executing standard Tree evaluation (Temps: 1.0 and {default_temp})...")
                    class_trees = build_class_trees(datapoints=train_id_datapoints, tree_type="ball", leaf_size=40)

                    for t in [1.0, default_temp]:
                        fn = partial(tree_distance_score, class_trees=class_trees, k_neighbors=50, temperature=t)
                        fn.__name__ = f"Tree||{t}"
                        eval_confidence_functions.append(fn)

            elif method_name == "edl":
                eval_confidence_functions = [edl_confidence]

            if method_name == "baseline" and is_tempscaling:
                logging.info("Learning optimal temperature on val set...")
                temp_scalar = TemperatureScaling(classifier)
                temperature = temp_scalar.learn_temperature(val_loader, device=DEVICE)
                logging.info(f"Learned optimal temperature: {temperature:.4f}")

            # 5. Execute Evaluation
            metrics_dict = evaluate_model(
                classifier=classifier,
                output_path=folder_path,
                test_loader=test_loader,
                device=DEVICE,
                confidence_functions=eval_confidence_functions,
                hooked_model=eval_hooked_model,
                temperature=temperature,
                save_plots=False
            )

            if method_name == "distance_based":
                eval_hooked_model.remove_hook()

            base_mod_name = modification_name.replace("(train)", "").strip()

            # 6. Append results
            if metrics_dict:
                for conf_method, metrics in metrics_dict.items():
                    actual_conf_fn = conf_method
                    final_mod_name = base_mod_name

                    if "||" in conf_method:
                        actual_conf_fn, temp_val_str = conf_method.split("||")
                        temp_val = float(temp_val_str)

                        if temp_val != 1.0:
                            if final_mod_name and final_mod_name.lower() != "standard":
                                final_mod_name = f"{final_mod_name} Temp {temp_val}"
                            else:
                                final_mod_name = f"Temp {temp_val}"

                    if not final_mod_name:
                        final_mod_name = "Standard"

                    all_results.append({
                        "Configuration": config_folder,
                        "Method": method_name.capitalize(),
                        "Modification": final_mod_name,
                        "Conf_Fn": actual_conf_fn,
                        **metrics
                    })

        except Exception as e:
            logging.error(f"Failed on {config_folder}/{method_name}/{variant_folder}: {str(e)}")
            continue

    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(by=["Configuration", "Method", "Modification"])
        df.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"\nAll done! Final results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_master_sweep()