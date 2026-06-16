import inspect
import logging
import time
from functools import partial
from pathlib import Path
from typing import List, Callable, Dict, Any, Union, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import auc, roc_curve


def plot_predictions_confidence(output_path: Union[str, Path], all_conf: np.ndarray, all_corr: np.ndarray,
                                all_ood: np.ndarray, fn_name: str, bins: int = 50, save_plot: bool = True) -> None:
    if not save_plot:
        return

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    sort_idx = np.argsort(all_conf)[::-1]
    all_conf_sorted = all_conf[sort_idx]
    all_corr_sorted = all_corr[sort_idx]
    all_ood_sorted = all_ood[sort_idx]

    fig, ax_main = plt.subplots(figsize=(6, 4.5))
    indices = np.arange(len(all_conf_sorted))

    ax_main.plot(indices, all_conf_sorted, color='blue', label='Confidence', linewidth=2.5, zorder=3)

    bin_edges = np.linspace(0, len(indices), bins + 1)
    correct_counts, _ = np.histogram(indices[all_corr_sorted == 1], bins=bin_edges)
    incorrect_counts, _ = np.histogram(indices[all_corr_sorted == 0], bins=bin_edges)

    corr_id_counts, _ = np.histogram(indices[(all_corr_sorted == 1) & (all_ood_sorted == 0)], bins=bin_edges)
    incorr_id_counts, _ = np.histogram(indices[(all_corr_sorted == 0) & (all_ood_sorted == 0)], bins=bin_edges)
    incorr_ood_counts, _ = np.histogram(indices[(all_corr_sorted == 0) & (all_ood_sorted == 1)], bins=bin_edges)

    max_count = max(max(correct_counts), max(incorrect_counts))
    scale_factor = 0.15 / max_count if max_count > 0 else 1

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.9

    ax_main.bar(bin_centers, -corr_id_counts * scale_factor, width=width, bottom=1.0, color='forestgreen', alpha=0.4,
                label='Correct ID')
    ax_main.bar(bin_centers, incorr_id_counts * scale_factor, width=width, bottom=0.0, color='purple', alpha=0.7,
                label='Incorrect ID')
    ax_main.bar(bin_centers, incorr_ood_counts * scale_factor, width=width, bottom=(incorr_id_counts * scale_factor),
                color='red', alpha=0.4, label='Incorrect OOD')

    ax_main.set_ylim(-0.05, 1.05)
    ax_main.set_xlabel('Samples (sorted by confidence)', fontsize=13)
    ax_main.set_ylabel('Confidence', fontsize=13)
    ax_main.set_title('Confidence vs. Error Density', fontsize=14, fontweight='bold')
    ax_main.tick_params(axis='both', which='major', labelsize=11)
    ax_main.legend(loc='center left', bbox_to_anchor=(0.02, 0.35), fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_dir / f"{fn_name}.png", dpi=300)
    plt.close()


def compute_ece(conf: np.ndarray, corr: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = np.logical_and(conf > lo.item(), conf <= hi.item())
        prob_in_bin = in_bin.mean()
        if prob_in_bin.item() > 0:
            accuracy_in_bin = corr[in_bin].mean()
            avg_confidence_in_bin = conf[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


def compute_mce(conf: np.ndarray, corr: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = np.logical_and(conf > lo, conf <= hi)
        if in_bin.mean() > 0:
            accuracy_in_bin = corr[in_bin].mean()
            avg_confidence_in_bin = conf[in_bin].mean()
            mce = max(mce, float(np.abs(avg_confidence_in_bin - accuracy_in_bin)))
    return mce


def compute_rmsce(conf: np.ndarray, corr: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rmsce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = np.logical_and(conf > lo, conf <= hi)
        prob_in_bin = in_bin.mean()
        if prob_in_bin > 0:
            accuracy_in_bin = corr[in_bin].mean()
            avg_confidence_in_bin = conf[in_bin].mean()
            rmsce += (avg_confidence_in_bin - accuracy_in_bin) ** 2 * prob_in_bin
    return float(np.sqrt(rmsce))


def compute_mec(conf: np.ndarray, corr: np.ndarray) -> float:
    Ci = 2 * corr - 1
    min_conf = conf.min()
    max_conf = conf.max()
    norm_conf = (conf - min_conf) / (max_conf - min_conf + 1e-12)
    return float(np.mean(Ci * norm_conf))


def compute_AUC(output_path: Union[str, Path], conf: np.ndarray, corr: np.ndarray, fn_name: str,
                save_plot: bool = True) -> float:
    fpr, tpr, _ = roc_curve(corr, conf)
    roc_auc = auc(fpr, tpr)

    if save_plot:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve for Confidence Score")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(out_dir / f"{fn_name}_AUC.png")
        plt.close()

    return roc_auc


def compute_AURC_and_Threshold(output_path: Union[str, Path], conf: np.ndarray, corr: np.ndarray, fn_name: str,
                               target_tpr: float, save_plot: bool = True) -> Tuple[float, float, float]:
    conf_noisy = conf + np.random.uniform(0, 1e-12, size=conf.shape)

    sort_indices = np.argsort(conf_noisy)[::-1]
    sorted_corr = corr[sort_indices]

    n_samples = len(conf)
    total_correct = np.sum(sorted_corr)
    total_wrong = n_samples - total_correct

    errors = 1 - sorted_corr
    cumulative_errors = np.cumsum(errors)
    cumulative_correct = np.cumsum(sorted_corr)
    kept_samples = np.arange(1, n_samples + 1)

    coverage_arr = kept_samples / n_samples
    risk_arr = cumulative_errors / kept_samples
    tpr_arr = cumulative_correct / total_correct
    fpr_arr = cumulative_errors / total_wrong

    aurc_score = np.sum(risk_arr) / n_samples

    optimal_cumulative_errors = np.maximum(0, kept_samples - total_correct)
    optimal_risk_arr = optimal_cumulative_errors / kept_samples
    optimal_aurc = np.sum(optimal_risk_arr) / n_samples

    e_aurc_score = aurc_score - optimal_aurc

    target_idx = np.argmax(tpr_arr >= target_tpr)
    fpr_at_target = fpr_arr[target_idx]
    risk_at_target = risk_arr[target_idx]
    cov_at_target = coverage_arr[target_idx]

    if save_plot:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.plot(coverage_arr, risk_arr, label=f"AURC = {aurc_score:.4f}\nE-AURC = {e_aurc_score:.4f}", color='red')
        plt.plot(coverage_arr, optimal_risk_arr, label="Optimal Risk (Baseline)", color='gray', linestyle='dashed')
        plt.scatter([cov_at_target], [risk_at_target], color='blue', zorder=5,
                    label=f"95% TPR Point (Risk: {risk_at_target:.2f})")

        plt.xlabel("Coverage (Proportion of dataset kept)")
        plt.ylabel("Risk (Error Rate on kept subset)")
        plt.title(f"Risk-Coverage Curve: {fn_name}")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.savefig(out_dir / f"{fn_name}_AURC.png")
        plt.close()

    return float(aurc_score), float(e_aurc_score), float(fpr_at_target)


def compute_Coverage_at_Risk(conf: np.ndarray, corr: np.ndarray, target_risk: float = 0.10) -> Tuple[float, float]:
    sort_indices = np.argsort(conf)[::-1]
    sorted_conf = conf[sort_indices]
    sorted_corr = corr[sort_indices]

    n_samples = len(conf)
    cumulative_errors = 0

    best_coverage = 0.0
    best_threshold = 1.0

    for i in range(n_samples):
        if sorted_corr[i] == 0:
            cumulative_errors += 1

        kept_samples = i + 1
        current_risk = cumulative_errors / kept_samples
        current_coverage = kept_samples / n_samples

        if current_risk > target_risk:
            break

        best_coverage = current_coverage
        best_threshold = sorted_conf[i]

    return best_coverage, best_threshold


def compute_ACC_at_Coverage(conf: np.ndarray, corr: np.ndarray, target_coverage: float = 0.95) -> float:
    sort_indices = np.argsort(conf)[::-1]
    sorted_corr = corr[sort_indices]

    n_samples = len(conf)
    keep_count = int(n_samples * target_coverage)

    if keep_count == 0:
        return 0.0

    kept_correct = np.sum(sorted_corr[:keep_count])
    return float(kept_correct / keep_count)


def _call_confidence_fn(fn: Callable, logits: Optional[torch.Tensor] = None, embeddings: Optional[torch.Tensor] = None,
                        preds: Optional[np.ndarray] = None) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = {}
    if "logits" in params:
        kwargs["logits"] = logits
    if "embeddings" in params:
        kwargs["embeddings"] = embeddings
    if "predictions" in params:
        kwargs["predictions"] = preds
    if "preds" in params:
        kwargs["preds"] = preds

    return fn(**kwargs)


def evaluate_model(classifier: nn.Module,
                   output_path: Union[str, Path],
                   test_loader: Any = None,
                   device: str = 'cuda',
                   confidence_functions: List[Callable] = [],
                   temperature: float = 1.0,
                   hooked_model: Optional[nn.Module] = None,
                   save_plots: bool = True) -> Dict[str, Dict[str, Any]]:
    model = hooked_model if hooked_model is not None else classifier
    model.eval()

    all_logits = []
    all_preds = []
    all_targets = []
    all_embeddings = []
    sample_input = None

    out_dir = Path(output_path)
    if save_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

    start_inference = time.time()

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if sample_input is None:
                sample_input = images[0:1]

            try:
                logits, emb = model(images, return_embeddings=True)
                all_embeddings.append(emb.detach().cpu())
            except TypeError:
                logits = model(images)

            preds = torch.argmax(logits, dim=1)

            if labels.ndim == 1:
                targets = labels.clone()
            else:
                targets = torch.argmax(labels, dim=1)

            all_logits.append(logits.cpu())  # keep as tensor for convenience
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    end_inference = time.time()
    base_inference_time = end_inference - start_inference
    logging.info(f"Base model inference time for dataset: {base_inference_time:.4f}s")

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_corr = (all_preds == all_targets).astype(float)

    num_seen_classes = all_logits[0].shape[1]
    max_target_label = int(np.max(all_targets))
    total_classes = max(num_seen_classes, max_target_label + 1)

    all_targets_grouped = np.copy(all_targets)
    all_targets_grouped[all_targets_grouped >= num_seen_classes] = num_seen_classes

    num_unseen = np.sum(all_targets_grouped == num_seen_classes)
    num_seen = len(all_targets_grouped) - num_unseen

    all_ood = (all_targets >= num_seen_classes).astype(int)

    logging.info("\n" + "=" * 50)
    logging.info("=== EVALUATION STATISTICS ===")
    logging.info("=" * 50)
    logging.info(f"Base inference time : {base_inference_time:.4f}s")
    logging.info(f"Total test samples  : {len(all_targets)} (ID: {num_seen} | OOD: {num_unseen})")

    label_range = list(range(num_seen_classes))

    seen_mask = all_targets_grouped != num_seen_classes
    if seen_mask.any():
        seen_targets = all_targets_grouped[seen_mask]
        seen_preds = all_preds[seen_mask]

        seen_acc = metrics.accuracy_score(seen_targets, seen_preds)
        seen_precision, seen_recall, seen_f1, _ = metrics.precision_recall_fscore_support(
            seen_targets, seen_preds,
            average='weighted',
            labels=label_range,
            zero_division=0.0,
        )
        logging.info(f"ID-Only  -> Acc: {seen_acc:.4f} | Prec: {seen_precision:.4f} | Rec: {seen_recall:.4f} | F1: {seen_f1:.4f}")

    # Accuracy of both seen and unseen samples
    acc = metrics.accuracy_score(all_targets_grouped, all_preds)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        all_targets_grouped, all_preds,
        average='weighted',
        labels=label_range + [num_seen_classes],
        zero_division=0.0,
    )
    logging.info(f"Combined -> Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")

    all_logits = torch.cat(all_logits, dim=0)
    all_logits = all_logits / all_logits.new_tensor(temperature)

    if len(all_embeddings) > 0:
        all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()  # [N, D] CPU tensor
    else:
        all_embeddings = None

    # Computation + size
    num_params = sum(p.numel() for p in classifier.parameters())
    base_size_MB = num_params * 4 / (1024 ** 2)

    # --- Metrics: FLOPS ---
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(sample_input,), verbose=False)
    except ImportError:
        flops = "N/A"
        logging.warning("Install 'thop' (pip install thop) to calculate FLOPs automatically.")

    if save_plots:
        cm_label_range = list(range(total_classes))
        cm = metrics.confusion_matrix(
            all_targets, all_preds,
            labels=cm_label_range,
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot(ax=ax, cmap='Blues')
        plt.tight_layout()
        plt.savefig(f"{output_path}/CM.png")
        plt.show()

    # logging.info(f"Model size (MB): {model_size_MB:.2f}")
    # logging.info(f"Confusion Matrix:\n{cm}")

    if len(confidence_functions) == 0:
        logging.warning("No confidence functions provided. Exiting evaluation.")
        return

    # --- HUGE APPENDIX TABLE HEADER ---
    latex_header = "Method & ECE & MCE & RMSCE & MEC & AUC & AURC & E-AURC & ACC@95 & FPR@95 & Coverage & FLOPs & Params & Mem (MB) & Time (s) \\\\"
    logging.info("LaTeX Table Header:")
    logging.info(latex_header)

    results_dict = {}

    for fn in confidence_functions:
        logging.info(f"--Confidence method: {fn.__name__}--")

        # Start total size with the base model size
        total_size_MB = base_size_MB
        total_params = num_params
        total_flops = flops

        fn_kwargs = {}
        if isinstance(fn, partial):
            fn_kwargs = fn.keywords

        # --- A. CONFNET OVERHEAD ---
        if "confnet" in fn_kwargs:
            confnet_model = fn_kwargs["confnet"]
            cn_params = sum(p.numel() for p in confnet_model.parameters())
            total_params += cn_params
            total_size_MB += (cn_params * 4) / (1024 ** 2)

            # Calculate extra FLOPs for the ConfNet forward pass
            if total_flops != "N/A":
                try:
                    from thop import profile
                    if fn_kwargs.get("use_features", False) and all_embeddings is not None:
                        dummy_in = torch.tensor(all_embeddings[0:1]).to(device)
                    else:
                        dummy_in = all_logits[0:1].to(device)
                    cn_flops, _ = profile(confnet_model, inputs=(dummy_in,), verbose=False)
                    total_flops += cn_flops
                except Exception:
                    pass

        # --- B. DISTANCE-BASED MEMORY OVERHEAD ---
        stored_vectors = 0
        emb_dim = all_embeddings.shape[1] if all_embeddings is not None else 1280

        try:
            if "centroids" in fn_kwargs:
                stored_vectors = fn_kwargs["centroids"].shape[0]
                emb_dim = fn_kwargs["centroids"].shape[1]

            elif "class_trees" in fn_kwargs:
                for c, tree_item in fn_kwargs["class_trees"].items():
                    # Unwrap the tree if it is hidden inside a dictionary
                    actual_tree = tree_item["tree"] if isinstance(tree_item, dict) else tree_item
                    stored_vectors += actual_tree.data.shape[0]
                    emb_dim = actual_tree.data.shape[1]

            elif "global_tree_dict" in fn_kwargs:
                tree_item = fn_kwargs["global_tree_dict"]
                # Unwrap the global tree
                actual_tree = tree_item["tree"] if isinstance(tree_item, dict) else tree_item
                stored_vectors = actual_tree.data.shape[0]
                emb_dim = actual_tree.data.shape[1]

            if stored_vectors > 0:
                # Memory = Num Vectors * Dimensions * 4 Bytes (Float32)
                memory_bank_MB = (stored_vectors * emb_dim * 4) / (1024 ** 2)
                total_size_MB += memory_bank_MB
                logging.info(f"Distance Overhead: Storing {stored_vectors} vectors -> +{memory_bank_MB:.2f} MB")

        except Exception as e:
            logging.warning(f"Could not calculate distance memory overhead for {fn.__name__}: {e}")

        # 3. Execute the confidence function
        start_conf = time.time()
        conf_output = _call_confidence_fn(fn, logits=all_logits, embeddings=all_embeddings, preds=all_preds)
        end_conf = time.time()

        # Calculate totals
        conf_time = end_conf - start_conf
        total_time_s = base_inference_time + conf_time

        if isinstance(conf_output, torch.Tensor):
            all_conf = conf_output.detach().cpu().numpy()
        else:
            all_conf = np.array(conf_output)

        if not np.all(all_conf >= 0.0) or not np.all(all_conf <= 1.0):
            logging.info(f"!Confidence scores out of bounds [0, 1] for {fn.__name__}, skipping evaluation!")
            continue

        plot_predictions_confidence(output_path, all_conf, all_corr, all_ood, fn.__name__, save_plot=save_plots)

        # --- Computations ---
        ece = compute_ece(all_conf, all_corr)
        mce = compute_mce(all_conf, all_corr)
        rmsce = compute_rmsce(all_conf, all_corr)
        mec = compute_mec(all_conf, all_corr)
        auc = compute_AUC(output_path, all_conf, all_corr, fn.__name__, save_plot=save_plots)
        aurc, e_aurc, fpr_at_95 = compute_AURC_and_Threshold(output_path, all_conf, all_corr, fn.__name__, 0.95, save_plot=save_plots)
        acc_95 = compute_ACC_at_Coverage(all_conf, all_corr, target_coverage=0.95)
        best_coverage, best_threshold = compute_Coverage_at_Risk(all_conf, all_corr, target_risk=0.10)
        coverage_pct = best_coverage * 100

        # --- LaTeX Formatting ---
        # Formats floats to 4 decimal places, size to 2 decimal places.
        def format_metric(val):
            if isinstance(val, str):
                return val
            if val >= 1e6:
                return f"{val / 1e6:.2f}M"
            elif val >= 1e3:
                return f"{val / 1e3:.2f}K"
            return str(val)

        # Use the dynamically updated totals here!
        flops_str = format_metric(total_flops)
        params_str = format_metric(total_params)

        logging.info(f"\n--- Results: {fn.__name__} ---")
        logging.info(f"ECE:   {ece:.4f}  | MCE:    {mce:.4f}  | RMSCE:  {rmsce:.4f}  | MEC:  {mec:.4f}")
        logging.info(f"AUC:   {auc:.4f}  | AURC:   {aurc:.4f}  | E-AURC: {e_aurc:.4f}")
        logging.info(f"ACC95: {acc_95:.4f}  | FPR95:  {fpr_at_95:.4f}  | Cov:    {coverage_pct:.2f}%")
        logging.info(f"FLOPs: {flops_str:<7} | Params: {params_str:<7} | Size:   {total_size_MB:.2f}MB | Time: {total_time_s:.4f}s")

        # LaTeX Row Output
        latex_row = (f"{fn.__name__.replace('_', '\\_')} & {ece:.4f} & {mce:.4f} & {rmsce:.4f} & {mec:.4f} & "
                     f"{auc:.4f} & {aurc:.4f} & {e_aurc:.4f} & {acc_95:.4f} & {fpr_at_95:.4f} & {coverage_pct:.2f}\\% & "
                     f"{flops_str} & {params_str} & {total_size_MB:.2f} & {total_time_s:.4f} \\\\")

        logging.info("LATEX TABLE OUTPUT")
        logging.info(latex_row)

        results_dict[fn.__name__] = {
            "ACC": acc,
            "ECE": ece,
            "MCE": mce,
            "RMSCE": rmsce,
            "MEC": mec,
            "AUC": auc,
            "AURC": aurc,
            "E-AURC": e_aurc,
            "ACC@95": acc_95,
            "FPR@95": fpr_at_95,
            "Cov": coverage_pct,
            "FLOPs": flops_str,
            "Params": params_str,
            "Size (MB)": total_size_MB,
            "Time (s)": total_time_s,
        }

    print("--------------------------\n")

    return results_dict
