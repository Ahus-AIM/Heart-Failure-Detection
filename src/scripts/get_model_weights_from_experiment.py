import csv
import os
import re
import shutil

HOME = os.path.expanduser("~")
EXPERIMENTS_DIR = os.path.join(HOME, "ray_results")
WEIGHTS_DIR = os.path.join("sandbox", "weights")
VAL_METRIC = "val_aucroc"


def get_val_fold_from_dirname(dirname: str) -> int | None:
    m = re.search(r"val_fold=(\d+)", dirname)
    return int(m.group(1)) if m else None


def parse_progress_csv(trial_dir: str, val_metric: str = VAL_METRIC) -> tuple[int | None, float | None]:
    progress_path = os.path.join(trial_dir, "progress.csv")
    if not os.path.isfile(progress_path):
        return None, None
    best_score: float | None = None
    best_epoch: int | None = None
    with open(progress_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if val_metric in row and row[val_metric]:
                try:
                    score = float(row[val_metric])
                    epoch = int(row["training_iteration"])
                    if best_score is None or score > best_score:
                        best_score = score
                        best_epoch = epoch
                except Exception:
                    continue
    return best_epoch, best_score


def find_checkpoint_dir(trial_dir: str, target_epoch: int) -> tuple[str | None, int | None]:
    checkpoints = [
        d for d in os.listdir(trial_dir) if os.path.isdir(os.path.join(trial_dir, d)) and d.startswith("checkpoint_")
    ]
    available_epochs: list[int] = []
    for d in checkpoints:
        m = re.match(r"checkpoint_(\d+)", d)
        if m:
            available_epochs.append(int(m.group(1)))
    if not available_epochs:
        return None, None
    available_epochs.sort()
    # Try exact match first
    if target_epoch in available_epochs:
        chosen_epoch = target_epoch
    else:
        # Use the highest available epoch less than target, else lowest available
        lower_epochs = [ep for ep in available_epochs if ep < target_epoch]
        if lower_epochs:
            chosen_epoch = max(lower_epochs)
        else:
            chosen_epoch = min(available_epochs)
    # Find the directory for the chosen epoch
    for d in checkpoints:
        m = re.match(r"checkpoint_(\d+)", d)
        if m and int(m.group(1)) == chosen_epoch:
            return os.path.join(trial_dir, d), chosen_epoch
    return None, None


def copy_checkpoint_flat(checkpoint_path: str, dest_dir: str, prefix: str) -> None:
    """
    Copy all files in checkpoint_path to dest_dir with filenames prefix+basename.
    If the checkpoint_path itself is a file, just copy and add prefix.
    """
    if os.path.isdir(checkpoint_path):
        files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f))]
        if not files:
            print(f"No files to copy in checkpoint dir: {checkpoint_path}")
            return
        for f in files:
            src_file = os.path.join(checkpoint_path, f)
            dest_file = os.path.join(dest_dir, f"{prefix}.pt")
            shutil.copy2(src_file, dest_file)
    elif os.path.isfile(checkpoint_path):
        basename = os.path.basename(checkpoint_path)
        dest_file = os.path.join(dest_dir, f"{prefix}_{basename}")
        shutil.copy2(checkpoint_path, dest_file)
        print(f"  -> {dest_file}")
    else:
        print(f"Unknown checkpoint path type: {checkpoint_path}")


def main() -> None:
    experiments = sorted(os.listdir(EXPERIMENTS_DIR))
    print("\nAvailable experiments (last 5):")
    for i, exp in enumerate(experiments[-5:], start=len(experiments) - 5):
        print(f"{i}: {exp}")
    selected = input("Select experiment by number, name, or type 'latest': ").strip()
    if selected == "latest":
        selected = experiments[-1]
    elif selected.isdigit() and int(selected) in range(len(experiments)):
        selected = experiments[int(selected)]
    elif selected not in experiments:
        print("Invalid experiment.")
        return

    exp_path = os.path.join(EXPERIMENTS_DIR, selected)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Group trial dirs by val_fold
    val_fold_to_trialdirs: dict[int, list[str]] = {}
    for trial_dir in os.listdir(exp_path):
        trial_dir_path = os.path.join(exp_path, trial_dir)
        if not os.path.isdir(trial_dir_path):
            continue
        val_fold = get_val_fold_from_dirname(trial_dir)
        if val_fold is not None:
            val_fold_to_trialdirs.setdefault(val_fold, []).append(trial_dir_path)

    if not val_fold_to_trialdirs:
        print("No val_folds found in any trial dir.")
        return

    # For each val_fold, find the best trial+epoch based on metric
    for val_fold, trial_dirs in val_fold_to_trialdirs.items():
        best_trial: str | None = None
        best_epoch: int | None = None
        best_score: float | None = None
        for trial_dir in trial_dirs:
            epoch, score = parse_progress_csv(trial_dir)
            if epoch is not None and (best_score is None or (score is not None and score > best_score)):
                best_trial = trial_dir
                best_epoch = epoch
                best_score = score

        if best_trial is None or best_epoch is None or best_score is None:
            print(f"No progress.csv with metric for val_fold={val_fold}")
            continue

        checkpoint_path, actual_epoch = find_checkpoint_dir(best_trial, best_epoch)
        if not checkpoint_path or actual_epoch is None:
            print(f"\nNo checkpoints found in {best_trial}!")
            continue

        prefix = f"{selected}_val_fold_{val_fold}_checkpoint_{actual_epoch:06d}"

        print(f"Copying checkpoint files for val_fold={val_fold} (best score {best_score:.4f}, epoch {actual_epoch}):")
        copy_checkpoint_flat(checkpoint_path, WEIGHTS_DIR, prefix)

    print("Done.")


if __name__ == "__main__":
    main()
