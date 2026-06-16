import subprocess
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(dataset: str, config: int, oe: float, exp_name: str, extra_args: List[str], eval_only: bool = False, parent_exp_name: Optional[str] = None) -> None:
    """Executes main.py with the defined matrix of arguments for distance_based method."""
    epochs = "20" if config == 1 else "50"

    cmd = [
        "python", "main.py",
        "--dataset", dataset,
        "--class-config", str(config),
        "--oe", str(oe),
        "--exp-name", exp_name,
        "--method", "distance_based",
        "--num-epochs", epochs,
    ]

    if eval_only:
        ckpt_folder = parent_exp_name if parent_exp_name else exp_name
        dist_ckpt = PROJECT_ROOT / f"../output/{dataset}/config_{config}/distance_based/{epochs}epochs{ckpt_folder}/checkpoint.pt"

        if not dist_ckpt.exists():
            print(f"Skipping {exp_name}: Checkpoint not found at {dist_ckpt.resolve()}")
            return

        cmd.extend(["--checkpoint-path", str(dist_ckpt)])
    else:
        unseen_text = f"_unseen{oe:.2f}" if oe > 0 else ""
        baseline_ckpt = PROJECT_ROOT / f"../output/{dataset}/config_{config}/baseline/{epochs}epochs{unseen_text}/checkpoint.pt"

        if baseline_ckpt.exists():
            cmd.extend(["--checkpoint-path", str(baseline_ckpt)])
        else:
            print(f"Warning: Baseline checkpoint not found at {baseline_ckpt.resolve()}. Training from scratch.")

        cmd.append("--train")

    cmd.extend(extra_args)

    print(f"\n{'=' * 80}")
    print(f"LAUNCHING: {dataset} | Config: {config} | OE: {oe} | Exp: {exp_name}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR in experiment {exp_name}: {e}")


def run_all() -> None:
    datasets = ["Cifar-100", "OhioSmallAnimals"]
    configs = [1, 2, 3]
    oes = [0.0, 0.5]

    OPT_ALPHA_NORM = "0.04"
    OPT_MARGIN_NORM = "1.4"
    OPT_ALPHA_UNNORM = "0.12"
    OPT_MARGIN_UNNORM = "5.5"

    experiments = {
        "_norm_(train)": ["--dist-l2-norm", "--dist-type", "kmeans", "--dist-margin", OPT_MARGIN_NORM, "--dist-alpha", OPT_ALPHA_NORM],
        "_(train)": ["--dist-type", "kmeans", "--dist-margin", OPT_MARGIN_UNNORM, "--dist-alpha", OPT_ALPHA_UNNORM]
    }

    for dataset in datasets:
        for config in configs:
            for oe in oes:
                full_exp_name = f"_unseen{oe:.2f}_norm_(train)"
                run_experiment(dataset, config, oe, full_exp_name, experiments["_norm_(train)"])

                if dataset == "Cifar-100" and config == 2:
                    full_exp_name = f"_unseen{oe:.2f}_(train)"
                    run_experiment(dataset, config, oe, full_exp_name, experiments["_(train)"])

    print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY. Check your output folder!")


def test_one() -> None:
    dataset = "OhioSmallAnimals"
    config = 2
    oe = 0.50

    unseen_text = f"_unseen{oe:.2f}" if oe > 0 else ""
    exp_name = f"{unseen_text}_norm_(train)"
    TRAIN = False

    extra_args = [
        "--dist-l2-norm",
        "--dist-margin", "1.4",
        "--dist-alpha", "0.1",
        "--dist-type", "tree",
        "--dist-clusters", "16"
    ]

    print(f"\n--- RUNNING SINGLE DISTANCE TEST: {exp_name} ---")
    run_experiment(dataset, config, oe, exp_name, extra_args, eval_only=not TRAIN)


def evaluate_all() -> None:
    datasets = ["Cifar-100", "OhioSmallAnimals"]
    configs = [1, 2, 3]
    oes = [0.0, 0.5]

    for dataset in datasets:
        for config in configs:
            for oe in oes:
                unseen_text = f"_unseen{oe:.2f}" if oe > 0 else ""
                parent_norm = f"{unseen_text}_norm_(train)"

                dist_methods = ["tree", "kmeans", "global"] if (dataset == "Cifar-100" and config == 2) else ["tree"]
                temps = ["0.1"] if config == 1 else ["0.4"]

                for method in dist_methods:
                    for temp in temps:
                        eval_name = f"{unseen_text}_eval_norm_{method}_temp{temp}"
                        extra_args = ["--dist-l2-norm", "--dist-type", method, "--dist-temp", temp]
                        run_experiment(dataset, config, oe, eval_name, extra_args, eval_only=True, parent_exp_name=parent_norm)

                if dataset == "Cifar-100" and config == 2:
                    parent_unnorm = f"{unseen_text}_(train)"
                    for method in dist_methods:
                        for temp in temps:
                            eval_name = f"{unseen_text}_eval_unnorm_{method}_temp{temp}"
                            extra_args = ["--dist-type", method, "--dist-temp", temp]
                            run_experiment(dataset, config, oe, eval_name, extra_args, eval_only=True, parent_exp_name=parent_unnorm)

    print("\nALL EVALUATIONS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    test_one()
    # run_all()
    # evaluate_all()