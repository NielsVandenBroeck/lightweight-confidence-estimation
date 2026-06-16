import subprocess
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(dataset: str, config: int, oe: float, exp_name: str, extra_args: List[str],
                   eval_only: bool = False) -> None:
    """Executes main.py with the defined matrix of arguments."""
    epochs = "20" if config == 1 else "50"

    cmd = [
        "python", "main.py",
        "--dataset", dataset,
        "--class-config", str(config),
        "--oe", str(oe),
        "--exp-name", exp_name,
        "--method", "confnet",
        "--num-epochs", epochs,
    ]

    if eval_only:
        confnet_ckpt = PROJECT_ROOT / f"../output/{dataset}/config_{config}/confnet/{epochs}epochs{exp_name}/checkpoint.pt"
        if not confnet_ckpt.exists():
            print(f"Skipping {exp_name}: Checkpoint not found at {confnet_ckpt.resolve()}")
            return

        cmd.extend(["--checkpoint-path", str(confnet_ckpt)])
    else:
        unseen_text = f"_unseen{oe:.2f}" if oe > 0 else ""
        baseline_ckpt = PROJECT_ROOT / f"../output/{dataset}/config_{config}/baseline/{epochs}epochs{unseen_text}/checkpoint.pt"

        if baseline_ckpt.exists():
            cmd.extend(["--checkpoint-path", str(baseline_ckpt)])
        else:
            print(f"Warning: Baseline checkpoint not found at {baseline_ckpt.resolve()}.")

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
    configs = [2, 3]
    oe = 0.5

    experiments_no_oe = {
        "": ["--conf-input", "logits", "--conf-sort"],
        "_BCE": ["--conf-input", "logits", "--conf-sort", "--bce-loss"],
        "_embeddings": ["--conf-input", "features", "--conf-hidden", "1280", "64"],
        "_embeddings_unfrozen": ["--conf-input", "features", "--conf-hidden", "1280", "64", "--unfreeze"],
        "_embeddings_BCE": ["--conf-input", "features", "--conf-hidden", "1280", "64", "--bce-loss"],
    }

    experiments_with_oe = {
        "_embeddings": ["--conf-input", "features", "--conf-hidden", "1280", "64"],
    }

    for dataset in datasets:
        for config in configs:
            for exp_name, extra_args in experiments_no_oe.items():
                run_experiment(dataset, config, 0.0, exp_name, extra_args)

            for exp_name, extra_args in experiments_with_oe.items():
                full_exp_name = f"_unseen{oe:.2f}{exp_name}"
                run_experiment(dataset, config, oe, full_exp_name, extra_args)

    print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY. Check your output folder!")


def test_one() -> None:
    dataset = "OhioSmallAnimals"
    config = 2
    oe = 0.50
    TRAIN = False

    unseen_text = f"_unseen{oe:.2f}" if oe > 0 else ""
    exp_name = f"{unseen_text}_embeddings"

    CONF_FREEZE = True
    CONF_INPUT = "features"
    CONF_SORT = False
    CONF_HIDDEN = [1280, 64]
    CONF_BCE_LOSS = False

    if not CONF_SORT and CONF_INPUT == "logits":
        exp_name = f"{unseen_text}_nosort"

    extra_args = ["--conf-input", CONF_INPUT]
    if CONF_SORT:
        extra_args.append("--conf-sort")
    if not CONF_FREEZE:
        extra_args.append("--unfreeze")
    if CONF_BCE_LOSS:
        extra_args.append("--bce-loss")
    if CONF_HIDDEN:
        extra_args.append("--conf-hidden")
        extra_args.extend([str(size) for size in CONF_HIDDEN])

    print(f"\n--- RUNNING SINGLE TEST: {exp_name} ---")
    run_experiment(dataset, config, oe, exp_name, extra_args, eval_only=not TRAIN)


if __name__ == "__main__":
    test_one()
    # run_all()