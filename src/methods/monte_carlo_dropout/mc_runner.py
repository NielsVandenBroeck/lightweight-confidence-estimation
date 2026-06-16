import subprocess
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(dataset: str, config: int, oe: float, exp_name: str, dropout: float = 0.2, mc_passes: int = 10, eval_only: bool = False, parent_exp_name: Optional[str] = None) -> None:
    """Executes main.py with the defined matrix of arguments for mc_dropout method."""
    epochs = "20" if config == 1 else "50"

    cmd = [
        "python", "main.py",
        "--dataset", dataset,
        "--class-config", str(config),
        "--oe", str(oe),
        "--exp-name", exp_name,
        "--method", "mc_dropout",
        "--num-epochs", epochs,
        "--dropout", str(dropout),
        "--mc-passes", str(mc_passes)
    ]

    if eval_only:
        ckpt_folder = parent_exp_name if parent_exp_name else exp_name
        ckpt_path = PROJECT_ROOT / f"../output_new/{dataset}/config_{config}/mc_dropout/{epochs}epochs{ckpt_folder}/checkpoint.pt"

        if not ckpt_path.exists():
            print(f"Skipping {exp_name}: Checkpoint not found at {ckpt_path.resolve()}")
            return

        cmd.extend(["--checkpoint-path", str(ckpt_path)])
    else:
        cmd.append("--train")

    print(f"\n{'=' * 80}")
    print(f"LAUNCHING: {dataset} | Config: {config} | OE: {oe} | Drop: {dropout} | Passes: {mc_passes}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR in experiment {exp_name}: {e}")


def run_train_all() -> None:
    datasets = ["Cifar-100", "OhioSmallAnimals"]
    configs = [1, 2, 3]
    oes = [0.0, 0.5]
    mc_passes = 10

    for dataset in datasets:
        for config in configs:
            for oe in oes:
                oe_text = f"_unseen{oe:.2f}" if oe > 0 else ""

                base_drop = 0.5
                exp_name_base = f"{oe_text}_drop{base_drop}"
                run_experiment(dataset, config, oe, exp_name_base, dropout=base_drop, mc_passes=mc_passes, eval_only=False)

                if dataset == "OhioSmallAnimals" and config in [1, 2] and oe == 0.0:
                    for drop in [0.3, 0.7]:
                        exp_name_special = f"{oe_text}_drop{drop}"
                        run_experiment(dataset, config, oe, exp_name_special, dropout=drop, mc_passes=mc_passes, eval_only=False)

    print("\nALL TRAINING EXPERIMENTS COMPLETED SUCCESSFULLY.")


def evaluate_all() -> None:
    datasets = ["Cifar-100"]
    configs = [2]
    oes = [0.0]
    dropouts = [0.5]
    passes_to_test = [1, 5, 10, 20, 50]

    for dataset in datasets:
        for config in configs:
            for oe in oes:
                for drop in dropouts:
                    oe_text = f"_unseen{oe:.2f}" if oe > 0 else ""
                    parent_exp_name = f"{oe_text}_drop{drop}"

                    for p in passes_to_test:
                        eval_name = f"{parent_exp_name}_passes{p}"
                        run_experiment(dataset, config, oe, eval_name, dropout=drop, mc_passes=p, eval_only=True, parent_exp_name=parent_exp_name)

    print("\nALL EVALUATIONS COMPLETED SUCCESSFULLY.")


def test_one() -> None:
    dataset = "OhioSmallAnimals"
    config = 2
    oe = 0.65
    dropout = 0.5
    mc_passes = 10
    TRAIN = False

    oe_text = f"_unseen{oe:.2f}" if oe > 0 else ""
    base_exp_name = f"{oe_text}_drop{dropout}"

    print(f"\n--- RUNNING SINGLE MC DROPOUT TEST: {base_exp_name} ---")

    if TRAIN:
        run_experiment(dataset, config, oe, base_exp_name, dropout=dropout, eval_only=False)
    else:
        eval_name = f"{base_exp_name}_passes{mc_passes}"
        run_experiment(dataset, config, oe, eval_name, dropout=dropout, mc_passes=mc_passes, eval_only=True, parent_exp_name=base_exp_name)


if __name__ == "__main__":
    test_one()
    # run_train_all()
    # evaluate_all()