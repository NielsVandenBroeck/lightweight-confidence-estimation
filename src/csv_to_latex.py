import re
from pathlib import Path
from typing import Tuple, Union

import pandas as pd


def get_sort_key(display_mod: str) -> float:
    """Assigns a hierarchical weight to sort modifications logically."""
    mod = display_mod.lower()
    score = 0.0

    # 1. Base Level (The Simplest Models)
    if mod == "standard":
        score = 0
    elif "maxprob" in mod and "oe" not in mod and "temp" not in mod:
        score = 1
    elif "wdf" in mod and "oe" not in mod and "temp" not in mod:
        score = 2
    elif "temp scaling" in mod and "oe" not in mod:
        score = 3

    # 2. ConfNet Specific
    elif "bce" in mod:
        score = 10
    elif "embeddings" in mod and "unfrozen" not in mod:
        score = 15
    elif "unfrozen" in mod:
        score = 18

    # 3. MC Dropout Specific (Sorts naturally by number of passes)
    elif "passes" in mod:
        try:
            p = int(re.search(r'\((\d+) passes\)', mod).group(1))
            score = 20 + (p * 0.01)
        except AttributeError:
            score = 20
    elif "dropout" in mod:
        score = 25

    # 4. Distance Specific
    elif "norm" in mod and "temp" not in mod:
        score = 30
    elif "norm" in mod and "temp" in mod:
        score = 31
    elif "tree" in mod or "kmeans" in mod or "global" in mod:
        score = 32

    # 5. Outlier Exposure (Pushes everything to the bottom half)
    if "oe" in mod:
        score += 100
        try:
            # OE 0.10 comes before OE 0.50
            oe_val = float(re.search(r'oe\s+(\d+\.\d+)', mod).group(1))
            score += (oe_val * 10)
        except AttributeError:
            pass

    return score


def clean_names(method: str, mod_str: str, conf_fn_str: str) -> Tuple[str, str]:
    """Polishes raw CSV strings into clean, readable LaTeX headers."""

    # 1. Clean the Method Name
    method_map = {
        "Baseline": "Baseline",
        "Confnet": "ConfNet",
        "Crl": "CRL",
        "Mc_dropout": "MC Dropout",
        "Distance_based": "Distance-Based",
        "Edl": "EDL"
    }
    safe_method = method_map.get(method.capitalize(), method.capitalize())

    # 2. Clean the Modification string
    m = str(mod_str)
    m = m.replace("Outlier Exposure", "OE")
    m = m.replace("tempscaling", "Temp Scaling")
    m = m.replace("embeddings", "Embeddings")
    m = m.replace("unfrozen", "Unfrozen")
    m = m.replace("_", " ")

    # Fix spacing issues from the CSV (e.g., "OE 0.50Embeddings" -> "OE 0.50 Embeddings")
    m = re.sub(r'(\d+\.\d+)([A-Za-z])', r'\1 \2', m).strip()

    # 3. Clean the Confidence Function string
    cfn = str(conf_fn_str).replace("_confidence", "").replace("_score", "")
    cfn = cfn.replace("maxprob", "MaxProb").replace("wdf", "WDF")
    cfn = cfn.replace("tree", "Tree").replace("kmeans", "K-Means").replace("global", "Global")

    # Hide the Conf_Fn entirely if it is redundant
    if cfn.lower() in method.lower() or cfn.lower() == "confnet":
        cfn = ""

    # 4. Combine smartly
    if m.lower() == "standard" or m == "":
        display_mod = cfn if cfn else "Standard"
    else:
        display_mod = f"{m} ({cfn})" if cfn else m

    # LaTeX escape
    return safe_method, display_mod.replace('%', '\\%').replace('_', '\\_')


def generate_latex_table(csv_path: Union[str, Path] = "../output/Cifar-100/thesis_results_OhioSmallAnimals.csv") -> None:
    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(f"Error: Could not find {csv_file}")
        return

    df = pd.read_csv(csv_file)

    config_titles = {
        "config_1": "Configuration 1: Broad (10 ID / 5 OOD)",
        "config_2": "Configuration 2: Easy (2 ID / 1 OOD)",
        "config_3": "Configuration 3: Hard (2 ID / 1 OOD)"
    }

    metrics = [
        "ECE", "RMSCE", "MEC", "AUC", "E-AURC", "ACC@95", "FPR@95",
        "Cov", "FLOPs", "Params", "Size (MB)", "Time (s)"
    ]

    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{ll " + "c" * len(metrics) + "}",
        "\\toprule",
        "\\textbf{Method} & \\textbf{Modification} & " + " & ".join(f"\\textbf{{{m}}}" for m in metrics) + " \\\\",
        "& & ($\\downarrow$) & ($\\downarrow$) & ($\\uparrow$) & ($\\uparrow$) & ($\\downarrow$) & ($\\uparrow$) & ($\\downarrow$) & ($\\uparrow$) & ($\\downarrow$) & ($\\downarrow$) & ($\\downarrow$) & ($\\downarrow$) \\\\"
    ]

    configs = sorted(df['Configuration'].unique())
    for config in configs:
        title = config_titles.get(config, f"Configuration: {config}")
        latex.append("\\midrule")
        latex.append(f"\\multicolumn{{{len(metrics) + 2}}}{{c}}{{\\textbf{{{title}}}}} \\\\")
        latex.append("\\midrule")

        config_df = df[df['Configuration'] == config]
        methods = sorted(config_df['Method'].unique())

        for m_idx, method in enumerate(methods):
            method_df = config_df[config_df['Method'] == method]
            n_rows = len(method_df)

            row_data = []
            for _, row in method_df.iterrows():
                raw_mod = str(row.get('Modification', ''))
                raw_conf_fn = str(row.get('Conf_Fn', ''))
                safe_method, display_mod = clean_names(method, raw_mod, raw_conf_fn)
                row_data.append((row, safe_method, display_mod))

            row_data.sort(key=lambda x: (get_sort_key(x[2]), x[2]))

            for i, (row, safe_method, display_mod) in enumerate(row_data):
                row_vals = []
                for m in metrics:
                    val = row.get(m, "N/A")

                    if pd.isna(val) or val == "N/A":
                        row_vals.append("-")
                    elif isinstance(val, (int, float)):
                        if m == "Cov":
                            row_vals.append(f"{val / 100.0:.4f}")
                        elif m in ["Size (MB)", "Time (s)"]:
                            row_vals.append(f"{val:.2f}")
                        else:
                            row_vals.append(f"{val:.4f}")
                    else:
                        row_vals.append(str(val).replace('%', '\\%'))

                val_str = " & ".join(row_vals)

                if i == 0:
                    latex.append(
                        f"\\multirow{{{n_rows}}}{{*}}{{\\textbf{{{safe_method}}}}} & {display_mod} & {val_str} \\\\")
                else:
                    latex.append(f" & {display_mod} & {val_str} \\\\")

            if m_idx < len(methods) - 1:
                latex.append("\\midrule")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Complete Evaluation Results across all configurations and methods.}",
        "\\label{tab:thesis_results_master}",
        "\\end{table*}"
    ])

    final_tex = "\n".join(latex)
    print(final_tex)


if __name__ == "__main__":
    generate_latex_table()