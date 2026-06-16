import subprocess
from pathlib import Path

# Dynamically find the project root (where main.py lives)
# Assuming this script is at /src/methods/baseline/baseline_runner.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(dataset: str, config: int, oe: float, exp_name: str, eval_only: bool = False) -> None:
    """Executes main.py with defined arguments for the baseline method."""
    epochs = "20" if config == 1 else "50"

    cmd = [
        "python", "main.py",
        "--dataset", dataset,
        "--class-config", str(config),
        "--oe", str(oe),
        "--exp-name", exp_name,
        "--method", "baseline",
        "--num-epochs", epochs,
        "--dropout", "0.5",
    ]

    if eval_only:
        # Resolve path relative to PROJECT_ROOT to maintain old behavior
        ckpt_path = PROJECT_ROOT / f"../output/{dataset}/config_{config}/baseline/{epochs}epochs{exp_name}/checkpoint.pt"

        if not ckpt_path.exists():
            print(f"Skipping {exp_name}: Checkpoint not found at {ckpt_path.resolve()}")
            return

        cmd.extend(["--checkpoint-path", str(ckpt_path)])
    else:
        cmd.append("--train")

    print(f"\n{'=' * 80}")
    print(f"LAUNCHING BASELINE: {dataset} | Config: {config} | OE: {oe} | Exp: {exp_name}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    try:
        # Execute the command from the Project Root so "python main.py" always works
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR in experiment {exp_name}: {e}")


def run_train_all() -> None:
    """Phase 1: Trains all baseline models with/without Outlier Exposure from scratch."""
    datasets = ["Cifar-100", "OhioSmallAnimals"]
    configs = [1, 2, 3]
    oes = [0.0, 0.5]

    for dataset in datasets:
        for config in configs:
            for oe in oes:
                oe_text = f"_unseen{oe:.2f}" if oe > 0 else ""
                exp_name = f"{oe_text}" if oe_text else ""
                run_experiment(dataset, config, oe, exp_name, eval_only=False)

    print("\nALL BASELINE TRAINING EXPERIMENTS COMPLETED.")


def test_one() -> None:
    """Quick test debugging toggle to make sure the pipeline runs smoothly."""
    dataset = "OhioSmallAnimals"
    config = 2
    oe = 0.50

    exp_name = f"_unseen{oe:.2f}"
    run_experiment(dataset, config, oe, exp_name, eval_only=False)


if __name__ == "__main__":
    test_one()
    # run_train_all()