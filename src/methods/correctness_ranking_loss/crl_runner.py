import subprocess
from pathlib import Path

# Dynamically find the project root (where main.py lives)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(dataset: str, config: int, oe: float, exp_name: str, eval_only: bool = False) -> None:
    """Executes main.py with defined arguments for the CRL method."""
    epochs = "20" if config == 1 else "50"

    cmd = [
        "python", "main.py",
        "--dataset", dataset,
        "--class-config", str(config),
        "--oe", str(oe),
        "--exp-name", exp_name,
        "--method", "crl",
        "--num-epochs", epochs,
    ]

    if eval_only:
        # Resolve path relative to PROJECT_ROOT
        ckpt_path = PROJECT_ROOT / f"../output/{dataset}/config_{config}/crl/{epochs}epochs{exp_name}/checkpoint.pt"

        # Fallback check for output_new just in case
        if not ckpt_path.exists():
            ckpt_path = PROJECT_ROOT / f"../output/{dataset}/config_{config}/crl/{epochs}epochs{exp_name}/checkpoint.pt"

        if not ckpt_path.exists():
            print(f"Skipping {exp_name}: Checkpoint not found at {ckpt_path.resolve()}")
            return

        cmd.extend(["--checkpoint-path", str(ckpt_path)])
    else:
        # Note: In main.py, CRL is currently set up to train from scratch.
        # If you ever want it to warm-start from the baseline, you would inject
        # the baseline checkpoint logic here (similar to dist_runner.py).
        cmd.append("--train")

    print(f"\n{'=' * 80}")
    print(f"LAUNCHING CRL: {dataset} | Config: {config} | OE: {oe} | Exp: {exp_name}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    try:
        # Execute the command from the Project Root so "python main.py" works
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR in experiment {exp_name}: {e}")


def run_train_all() -> None:
    """Phase 1: Trains all CRL models with/without Outlier Exposure."""
    datasets = ["Cifar-100", "OhioSmallAnimals"]
    configs = [1, 2, 3]
    oes = [0.0, 0.5]

    for dataset in datasets:
        for config in configs:
            for oe in oes:
                oe_text = f"_unseen{oe:.2f}" if oe > 0 else ""
                exp_name = f"{oe_text}" if oe_text else ""
                run_experiment(dataset, config, oe, exp_name, eval_only=False)

    print("\nALL CRL TRAINING EXPERIMENTS COMPLETED.")


def test_one() -> None:
    """Quick test debugging toggle to make sure the pipeline runs smoothly."""
    dataset = "OhioSmallAnimals"
    config = 2
    oe = 0.50
    TRAIN = False

    exp_name = f"_unseen{oe:.2f}" if oe > 0 else ""

    print(f"\n--- RUNNING SINGLE CRL TEST: {exp_name if exp_name else 'Standard'} ---")
    run_experiment(dataset, config, oe, exp_name, eval_only=not TRAIN)


if __name__ == "__main__":
    test_one()
    # run_train_all()