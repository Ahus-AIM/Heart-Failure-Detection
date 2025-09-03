import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils import log_step


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml as _yaml

    with open(path, "r", encoding="utf-8") as f:
        return _yaml.safe_load(f)


def load_catalogue(d_labitems_path: Path) -> pd.DataFrame:
    cat = pd.read_csv(d_labitems_path, dtype={"itemid": "Int64"})
    cat["label_norm"] = cat["label"].astype(str).str.strip().str.lower()
    return cat


def resolve_itemids(cat: pd.DataFrame, tests: List[str]) -> dict:
    itemids = {}
    tests_norm = [t.strip().lower() for t in tests]
    for t in tests_norm:
        match = cat[cat["label_norm"].str.contains(t, na=False)]
        if not match.empty:
            itemids[t] = match["itemid"].dropna().astype(int).tolist()
    return itemids


def load_labevents(labevents_path: Path, keep_itemids: set) -> pd.DataFrame:
    """
    Efficiently load only needed rows/columns from a very large labevents CSV.
    """
    keep_itemids = {int(x) for x in keep_itemids}
    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    dtypes = {
        "subject_id": "int32",
        "hadm_id": "Int32",
        "itemid": "int32",
        "valuenum": "float32",
    }
    engine = "c"

    chunks = []
    for chunk in pd.read_csv(
        labevents_path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["charttime"],
        chunksize=1_000_000,
        engine=engine,
    ):
        chunk = chunk[chunk["itemid"].isin(keep_itemids)]
        if chunk.empty:
            continue
        if "valueuom" in chunk.columns:
            chunk["valueuom"] = chunk["valueuom"].astype("category")
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["subject_id", "charttime"], kind="mergesort", ignore_index=True)
    return df


def load_ecg_links(wf_links_path: Path) -> pd.DataFrame:
    df = pd.read_csv(wf_links_path)
    time_col = None
    for c in ["charttime", "chart_time", "linkdt", "link_time", "datetime", "study_datetime", "study_time"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("Could not find a timestamp column in waveform_note_links.csv")
    df["ecg_time"] = pd.to_datetime(df[time_col])
    keep_cols = ["subject_id", "ecg_time"]
    for extra in ["recordname", "path", "study_id", "ecg_id", "note_id", "file"]:
        if extra in df.columns:
            keep_cols.append(extra)
    df = df[keep_cols].dropna(subset=["subject_id", "ecg_time"]).copy()
    return df.sort_values(["subject_id", "ecg_time"], kind="mergesort", ignore_index=True)


def link_labs_to_nearest_ecg(
    labs: pd.DataFrame,
    ecg_links: pd.DataFrame,
    *,
    require_ecg: bool = True,
    add_ecg_time: bool = False,
) -> pd.DataFrame:
    """
    For each lab row, find the closest ECG time for the same subject.
    If require_ecg=True, labs for subjects with no ECG are dropped.
    Set add_ecg_time=True to keep a 'nearest_ecg_time' column (useful for QA).
    """
    out_rows = []

    ecg_by_subj = {
        sid: grp["ecg_time"].values.astype("datetime64[ns]") for sid, grp in ecg_links.groupby("subject_id", sort=False)
    }

    for sid, labs_sub in labs.groupby("subject_id", sort=False):
        ecg_times = ecg_by_subj.get(sid)
        if ecg_times is None or ecg_times.size == 0:
            if require_ecg:
                continue
            else:
                for _, row in labs_sub.iterrows():
                    out = {
                        "subject_id": int(row["subject_id"]),
                        "itemid": int(row["itemid"]),
                        "charttime": row["charttime"],
                        "valuenum": float(row["valuenum"]) if pd.notna(row["valuenum"]) else np.nan,
                        "valueuom": row.get("valueuom", pd.NA),
                    }
                    if add_ecg_time:
                        out["nearest_ecg_time"] = pd.NaT
                    out_rows.append(out)
                continue

        # Vectorized-ish: for each lab time, find nearest ECG by absolute delta
        lab_times = labs_sub["charttime"].values.astype("datetime64[ns]")
        for i, (_, lab_row) in enumerate(labs_sub.iterrows()):
            lt = lab_times[i]
            j = int(np.argmin(np.abs(ecg_times - lt)))
            nearest = pd.to_datetime(ecg_times[j])

            out = {
                "subject_id": int(lab_row["subject_id"]),
                "itemid": int(lab_row["itemid"]),
                "charttime": pd.to_datetime(lt),
                "valuenum": float(lab_row["valuenum"]) if pd.notna(lab_row["valuenum"]) else np.nan,
                "valueuom": lab_row.get("valueuom", pd.NA),
            }
            if add_ecg_time:
                out["nearest_ecg_time"] = nearest
            out_rows.append(out)

    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path, help="YAML config file for preparing MIMIC-IV index")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    d_labitems = Path(cfg["paths"]["d_labitems"])
    labevents = Path(cfg["paths"]["labevents"])
    waveform_note_links = Path(cfg["paths"]["waveform_note_links"])
    out_path = Path(cfg["paths"]["out"]).expanduser()

    # Which lab(s) to resolve
    tests = ["NTproBNP"]

    with log_step("Resolving lab tests to itemids"):
        cat = load_catalogue(d_labitems)
        tests_map = resolve_itemids(cat, tests)
        if not tests_map:
            raise SystemExit("No matching tests found in d_labitems for requested 'tests'.")

    keep_ids = {iid for ids in tests_map.values() for iid in ids}

    with log_step("Loading and filtering lab events"):
        labs = load_labevents(labevents, keep_ids)

    with log_step("Loading ECG links"):
        ecg_links = load_ecg_links(waveform_note_links)

    with log_step("Linking labs to nearest ECG (lab-centric)"):
        linked = link_labs_to_nearest_ecg(labs, ecg_links, require_ecg=True, add_ecg_time=args.keep_ecg_time)

    with log_step("Selecting and ordering columns"):
        final_cols = ["subject_id", "itemid", "charttime", "valuenum", "valueuom"]
        if args.keep_ecg_time:
            final_cols.append("nearest_ecg_time")
        linked = linked[final_cols].sort_values(["subject_id", "charttime"], kind="mergesort", ignore_index=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)

    with log_step("Writing output"):
        linked.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} (rows={len(linked)})")


if __name__ == "__main__":
    main()
