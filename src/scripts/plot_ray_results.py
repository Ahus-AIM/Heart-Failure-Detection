import os

import matplotlib.pyplot as plt
import pandas as pd

HOME: str = os.path.expanduser("~")
EXPERIMENTS_DIR: str = os.path.join(HOME, "ray_results")
PLOTS_DIR: str = os.path.join("sandbox", "ray_analysis")
VAL_PREFIX: str = "val_"
TRAIN_PREFIX: str = "train_"


def list_experiments(base_dir: str) -> list[str]:
    experiments: list[str] = sorted(os.listdir(base_dir))
    print("\nAvailable experiments (last 5):")
    for i, exp in enumerate(experiments[-5:], start=len(experiments) - 5):
        print(f"{i}: {exp}")
    return experiments


def select_experiment(experiments: list[str]) -> str:
    while True:
        selection: str = input("Select experiment by number, name, or type 'latest': ").strip()
        if selection == "latest":
            return experiments[-1]
        if selection.isdigit() and int(selection) in range(len(experiments)):
            return experiments[int(selection)]
        if selection in experiments:
            return selection
        print("Invalid selection. Try again.")


def collect_results(
    ray_path: str,
    val_prefix: str = VAL_PREFIX,
    train_prefix: str = TRAIN_PREFIX,
) -> tuple[dict[str, dict[str, list[float]]], set[str]]:
    results: dict[str, dict[str, list[float]]] = {}
    col_names: set[str] = set()
    for root, dirs, files in os.walk(ray_path):
        for file in files:
            if file == "progress.csv":
                df = pd.read_csv(os.path.join(root, file))
                results[root] = {}
                for col in df.columns:
                    if col.startswith((val_prefix, train_prefix)):
                        results[root][col] = df[col].values.tolist()
                        col_names.add(col)
    return results, col_names


def extract_metric_names(
    col_names: set[str],
    val_prefix: str = VAL_PREFIX,
    train_prefix: str = TRAIN_PREFIX,
) -> set[str]:
    col_names_no_prefix: set[str] = set()
    for col in col_names:
        if col.startswith(val_prefix):
            col_names_no_prefix.add(col[len(val_prefix) :])
        elif col.startswith(train_prefix):
            col_names_no_prefix.add(col[len(train_prefix) :])
    return col_names_no_prefix


def plot_metrics(
    results: dict[str, dict[str, list[float]]],
    metric_names: set[str],
    outdir: str,
    val_prefix: str = VAL_PREFIX,
    train_prefix: str = TRAIN_PREFIX,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    for col in metric_names:
        fig, ax = plt.subplots(1, 2, figsize=(25, 10))
        for run, data in results.items():
            runname: str = run.split("/")[-1]  # [20:]
            runname = " ".join(runname.split("_")[5:-2])
            runname = "=".join(runname.split("=")[:])
            if f"{train_prefix}{col}" in data:
                ax[0].plot(data[f"{train_prefix}{col}"], label=runname)
            ax[0].set_title(f"Training {col}")
            ax[0].grid(True)
            ax[0].legend()
            if f"{val_prefix}{col}" in data:
                ax[1].plot(data[f"{val_prefix}{col}"], label=runname)
            ax[1].set_title(f"Validation {col}")
            ax[1].set_ylim(ax[0].get_ylim())
            ax[1].grid(True)
            ax[1].legend()
        plt.tight_layout()
        plot_path: str = os.path.join(outdir, f"{col}.png")
        plt.savefig(plot_path)
        print(f"Saved: {plot_path}")
        plt.close(fig)


def summarize_runs(
    results: dict[str, dict[str, list[float]]],
    metrics: list[str],
    val_prefix: str = VAL_PREFIX,
    train_prefix: str = TRAIN_PREFIX,
) -> pd.DataFrame:
    for metric in metrics:
        rows = []
        try:
            for run, data in results.items():
                runname: str = run.split("/")[-1]
                runname = " ".join(runname.split("_")[5:-2])
                runname = "=".join(runname.split("=")[:])
                params = {}
                for kv in runname.split(","):
                    if "=" in kv:
                        k, v = kv.split("=")
                        params[k.strip()] = v.strip()
                val_metric = data.get(f"{val_prefix}{metric}", [])
                train_metric = data.get(f"{train_prefix}{metric}", [])
                max_val = max(val_metric) if val_metric else float("-inf")
                max_train = max(train_metric) if train_metric else float("-inf")
                row = {
                    **params,
                    "max_val_metric": max_val,
                    "max_train_metric": max_train,
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            # Coerce columns to numeric for sorting
            df["max_val_metric"] = pd.to_numeric(df["max_val_metric"], errors="coerce")
            df["max_train_metric"] = pd.to_numeric(df["max_train_metric"], errors="coerce")
            # Now sort
            df = df.sort_values("max_val_metric", ascending=False)
            print(df.to_markdown(index=False))
        except Exception as e:
            print(f"Error summarizing runs for metric '{metric}': {e}")
            continue
    return df


def main() -> None:
    experiments: list[str] = list_experiments(EXPERIMENTS_DIR)
    if not experiments:
        print(f"No experiments found in {EXPERIMENTS_DIR}")
        return
    selected: str = select_experiment(experiments)
    ray_path: str = os.path.join(EXPERIMENTS_DIR, selected)
    print(f"\nAnalyzing experiment: {selected}")

    results, col_names = collect_results(ray_path)
    metric_names = extract_metric_names(col_names)

    print("\nMetrics found:")
    for m in sorted(metric_names):
        print(" ", m)
    plots_dir = os.path.join(PLOTS_DIR, selected)
    print(f"\nSaving plots to: {plots_dir}")

    plot_metrics(results, metric_names, plots_dir)
    summarize_runs(results, ["challenge_score", "topk_accuracy"])


if __name__ == "__main__":
    main()
