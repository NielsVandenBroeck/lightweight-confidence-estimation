import argparse
import logging
import os
import pprint
import random
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

from methods.confnet import ClassificationConfidenceNetwork, confnet_score, compute_confnet_loss, \
    compute_confnet_bce_loss
from methods.correctness_ranking_loss import CRLLoss
from methods.distance_based_confidence import (
    compute_datapoints, build_global_tree, build_class_kmeans,
    build_class_trees, HookedModel, global_distance_score,
    kmeans_distance_score, tree_distance_score,
    plot_class_kmeans, compute_distance_loss, plot_distance_distributions
)
from methods.evidential_deep_learning import EDLLoss, edl_confidence
from methods.monte_carlo_dropout import MCDropoutWrapper, mc_dropout_confidence
from methods.outlier_exposure import oe_loss
from methods.output_based_confidence import maxprob_confidence, wdf_confidence
from methods.temperature_scaling import TemperatureScaling
from src.data_processing.dataset import create_dataloaders
from src.evaluation import evaluate_model
from src.models import get_efficientnet
from src.train import Trainer

# Set strict random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet Model Pipeline")

    # Core Settings
    parser.add_argument('--root-path', type=str, default="../datasets", help='Path to the dataset folder')
    parser.add_argument('--dataset', type=str, choices=['Cifar-100', 'OhioSmallAnimals'], required=True,
                        help='Target dataset')
    parser.add_argument('-i', '--img-size', type=int, nargs=2, default=[224, 224],
                        help='Image dimensions: height width')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('-w', '--num-workers', type=int, default=4, help="Dataloader worker processes")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4, help='Weight Decay')
    parser.add_argument('-e', '--num-epochs', type=int, default=20, help='Total epochs')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to resume training checkpoint')
    parser.add_argument('--no-checkpoint-path', action='store_true', default=False, help='Overrides --checkpoint-path')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Execute training loop (otherwise eval only)')
    parser.add_argument('--method', type=str,
                        choices=['baseline', 'distance_based', 'confnet', 'crl', 'mc_dropout', 'edl'],
                        default='baseline')

    # Experiments
    parser.add_argument('--class-config', type=int, required=True, help="Class separation configuration (1, 2, or 3)")
    parser.add_argument('--oe', type=float, default=0.0, help="Outlier Exposure proportion")
    parser.add_argument('--exp-name', type=str, default="", help="Experiment suffix for output folder")

    # ConfNet Ablations
    parser.add_argument('--conf-input', type=str, choices=['logits', 'features'])
    parser.add_argument('--conf-hidden', type=int, nargs='*', help="e.g. 512 128 64")
    parser.add_argument('--conf-sort', action='store_true', help="Sort input probabilities")
    parser.add_argument('--conf_alpha', type=float, default=0.01, help="alpha param in loss")
    parser.add_argument('--unfreeze', action='store_true', help="Joint training with backbone")
    parser.add_argument('--bce-loss', action='store_true', help="Use BCE loss over ConfNet loss")

    # Distance-Based Ablations
    parser.add_argument('--dist-l2-norm', action='store_true', help="L2 Norm prior to clustering")
    parser.add_argument('--dist-type', type=str, choices=['global', 'tree', 'kmeans'], default='kmeans')
    parser.add_argument('--dist-clusters', type=int, default=16, help="KMeans clusters per class")
    parser.add_argument('--dist-k', type=int, default=50, help="K parameter for tree/global")
    parser.add_argument('--dist-margin', type=float, default=1.5, help="Contrastive loss margin")
    parser.add_argument('--dist-alpha', type=float, default=0.1, help="Distance loss weight")
    parser.add_argument('--dist-temp', type=float, default=1.0, help="Temperature scaling for density weights")

    # Monte-Carlo Ablations
    parser.add_argument('--dropout', type=float, default=0.2, help="MC Dropout rate")
    parser.add_argument('--mc-passes', type=int, default=10, help="MC Dropout forward passes")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DROPOUT = args.dropout
    MC_PASSES = args.mc_passes
    TEMPSCALING = False

    args.output_path = f"../output/{args.dataset}/config_{args.class_config}/{args.method}/{args.num_epochs}epochs{args.exp_name}"
    os.makedirs(args.output_path, exist_ok=True)
    args.root_path = os.path.join(args.root_path, args.dataset)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_path, "train.log")),
            logging.StreamHandler(),
        ]
    )

    logging.info("=" * 50)
    logging.info("=== INITIALIZING EXPERIMENT ===")
    logging.info("=" * 50)
    logging.info("Run Configuration:\n" + pprint.pformat(vars(args), indent=4))
    logging.info(f"Using device: {args.device}")

    # ==========================================
    # DATASET SETUP
    # ==========================================
    if args.class_config == 1:
        if args.dataset == "Cifar-100":
            classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark',
                       'trout']
            unseen_classes = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
        else:
            classes = ['Bird-Sparrow', 'Bird-Wren', 'Snake-Brownsnake', 'Snake-Gartersnake', 'Mouse-White', 'Shrew',
                       'Mink', 'Chipmunk', 'Frog', 'Lizard']
            unseen_classes = ['Bird-Yellowthroat', 'Snake-Massasauga', 'Mouse-Woodland', 'Vole', 'Weasel']

    elif args.class_config == 2:
        if args.dataset == "Cifar-100":
            classes, unseen_classes = ['bear', 'fox'], ['squirrel']
        else:
            classes, unseen_classes = ['Chipmunk', 'Vole'], ['Mouse-Woodland']

    elif args.class_config == 3:
        if args.dataset == "Cifar-100":
            classes, unseen_classes = ['shark', 'whale'], ['dolphin']
        else:
            classes, unseen_classes = ['Snake-Gartersnake', 'Snake-Massasauga'], ['Snake-Brownsnake']
    else:
        logging.error(f"Invalid Class config: {args.class_config}")
        exit(1)

    logging.info(f"Main classes: {classes} | Unseen classes: {unseen_classes}")

    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
        unseen_train_proportion=args.oe,
    )

    num_model_classes = len(classes)
    logging.info(f"Detected number of classes: {num_model_classes}")

    # ==========================================
    # MODEL & TRAINING STRATEGY SETUP
    # ==========================================
    classifier = get_efficientnet(version=0, num_classes=num_model_classes, is_pretrained=True,
                                  dropout_rate=DROPOUT).to(args.device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    custom_loss_fn = None
    conf_fn_for_val = None
    confnet = None

    if args.method == "confnet":
        FEATURE_DIM = 1280
        input_size = FEATURE_DIM if args.conf_input == "features" else num_model_classes
        is_linear = args.conf_hidden is None

        confnet = ClassificationConfidenceNetwork(
            input_dim=input_size,
            is_linear=is_linear,
            sort_input=args.conf_sort,
            hidden_channels=args.conf_hidden,
        ).to(args.device)

        if args.unfreeze:
            for param in classifier.parameters():
                param.requires_grad = False
            optimizer = optim.Adam(confnet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam([
                {'params': classifier.parameters(), 'lr': 1e-5, 'weight_decay': args.weight_decay},
                {'params': confnet.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}
            ])

        if args.bce_loss:
            custom_loss_fn = partial(compute_confnet_bce_loss, confnet=confnet,
                                     use_features=(args.conf_input == "features"))
        else:
            custom_loss_fn = partial(compute_confnet_loss, confnet=confnet,
                                     use_features=(args.conf_input == "features"), alpha=args.conf_alpha)

        conf_fn_for_val = partial(confnet_score, confnet=confnet, device=args.device,
                                  use_features=(args.conf_input == "features"))
        conf_fn_for_val.__name__ = "confnet_confidence"

    elif args.method == "crl":
        optimizer = optim.Adam(list(classifier.parameters()), lr=1e-5, weight_decay=args.weight_decay)
        custom_loss_fn = CRLLoss(b=32, crl_weight=0.1)
        conf_fn_for_val = maxprob_confidence

    elif args.method == "edl":
        custom_loss_fn = EDLLoss(num_classes=num_model_classes, annealing_steps=args.num_epochs / 2)
        conf_fn_for_val = edl_confidence

    elif args.method == "distance_based":
        custom_loss_fn = partial(compute_distance_loss, margin=args.dist_margin, alpha=args.dist_alpha,
                                 apply_l2_norm=args.dist_l2_norm, ood_label=num_model_classes)
        conf_fn_for_val = maxprob_confidence

    else:
        if args.oe > 0:
            custom_loss_fn = oe_loss
        conf_fn_for_val = maxprob_confidence

    # ==========================================
    # EXECUTION PIPELINE
    # ==========================================
    if args.train:
        logging.info("---- Training model ----")
        trainer = Trainer(classifier=classifier, optimizer=optimizer, device=args.device, output_path=args.output_path,
                          checkpoint_path=args.checkpoint_path, method=args.method, loss_fn=custom_loss_fn,
                          confnet=confnet)
        trainer.fit(train_loader, val_loader, args.num_epochs, conf_fn_for_val)

        best_ckpt_path = os.path.join(args.output_path, "checkpoint.pt")
        if os.path.exists(best_ckpt_path):
            checkpoint = torch.load(best_ckpt_path)
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            logging.info("Classifier weights loaded successfully!")
            if confnet is not None and 'confnet_state_dict' in checkpoint:
                confnet.load_state_dict(checkpoint['confnet_state_dict'])
                logging.info("ConfNet weights loaded successfully!")

    elif args.checkpoint_path is not None:
        logging.info(f"Loading checkpoint... {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        logging.info("Classifier weights loaded successfully!")
        if confnet is not None and 'confnet_state_dict' in checkpoint:
            confnet.load_state_dict(checkpoint['confnet_state_dict'])
            logging.info("ConfNet weights loaded successfully!")
    else:
        logging.error("No checkpoint or training instruction provided. Exiting.")
        exit(1)

    # ==========================================
    # EVALUATION STRATEGY
    # ==========================================
    logging.info("=" * 50)
    logging.info(f"=== EVALUATING MODEL: {args.method.upper()} ===")
    logging.info("=" * 50)

    eval_confidence_functions = []
    eval_hooked_model = None
    temperature = 1.0

    if args.method == "distance_based":
        eval_hooked_model = HookedModel(classifier).to(args.device)
        all_train_datapoints = compute_datapoints(eval_hooked_model, train_loader, args.device, keep_ood=True,
                                                  normalize=args.dist_l2_norm, ood_label=num_model_classes)

        train_id_datapoints = [(lbl, emb) for lbl, emb in all_train_datapoints if lbl != num_model_classes]
        train_ood_datapoints = [(lbl, emb) for lbl, emb in all_train_datapoints if lbl == num_model_classes]
        test_datapoints = compute_datapoints(eval_hooked_model, test_loader, args.device, keep_ood=True,
                                             normalize=args.dist_l2_norm, ood_label=num_model_classes)

        if args.dist_type == "kmeans":
            class_kmeans = build_class_kmeans(datapoints=train_id_datapoints, n_clusters_per_class=args.dist_clusters)
            plot_class_kmeans(train_id_datapoints, class_kmeans, test_datapoints, train_ood_datapoints,
                              num_model_classes, args.output_path)
            logging.info("Generating pairwise distance distributions...")
            plot_distance_distributions(test_datapoints, num_model_classes, args.output_path)

            centroids = np.array([class_kmeans[c]["centroids"].mean(axis=0) for c in sorted(class_kmeans.keys())])
            distance_confidence = partial(kmeans_distance_score, centroids=centroids, temperature=args.dist_temp)

        elif args.dist_type == "tree":
            class_trees = build_class_trees(datapoints=train_id_datapoints, tree_type="ball", leaf_size=40)
            distance_confidence = partial(tree_distance_score, class_trees=class_trees, k_neighbors=args.dist_k,
                                          temperature=args.dist_temp)

        elif args.dist_type == "global":
            global_tree_dict = build_global_tree(datapoints=train_id_datapoints, leaf_size=40)
            distance_confidence = partial(global_distance_score, global_tree_dict=global_tree_dict,
                                          k_neighbors=args.dist_k, temperature=args.dist_temp)

        distance_confidence.__name__ = "distance_confidence"
        eval_confidence_functions = [distance_confidence]

    elif args.method == "confnet":
        eval_confidence_functions = [conf_fn_for_val]

    elif args.method == "crl":
        eval_confidence_functions = [maxprob_confidence]

    elif args.method == "mc_dropout":
        eval_hooked_model = MCDropoutWrapper(classifier, num_passes=MC_PASSES)
        eval_confidence_functions = [mc_dropout_confidence]

    elif args.method == "edl":
        eval_confidence_functions = [edl_confidence]

    else:
        if TEMPSCALING:
            temp_scalar = TemperatureScaling(classifier)
            temperature = temp_scalar.learn_temperature(val_loader, device=args.device)
            logging.info(f"Learned optimal temperature: {temperature:.4f}")
        eval_confidence_functions = [maxprob_confidence, wdf_confidence]

    evaluate_model(
        classifier,
        args.output_path,
        test_loader=test_loader,
        device=args.device,
        confidence_functions=eval_confidence_functions,
        hooked_model=eval_hooked_model,
        temperature=temperature
    )

    if args.method == "distance_based":
        eval_hooked_model.remove_hook()

    logging.info("Done.")