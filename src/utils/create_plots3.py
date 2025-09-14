# src/utils/create_plots3.py
#!/usr/bin/env python3
"""
Plot three metrics for SMAC/SMACv2 logs, choosing the BEST run per model
based on the final test_battle_won_mean.

Folder layout (with run subfolders):
  ROOT/<sc_race>/<game>/<model>/<run>/info.json
e.g., results/sacred/10gen_protoss/5v5/hpn_qmix/1/info.json

For each (<sc_race>, <game>), and for the models given with --models, we:
  - pick the run whose FINAL test_battle_won_mean is highest,
  - plot:
      1) test_return_mean (fill from test_return_std of that run),
      2) test_battle_won_mean,
      3) Training_avg_time,
  - align x-ranges by truncating to the MIN last-step across the selected models
    for the metric being plotted,
  - save one PNG per metric per (<sc_race>, <game>) under:
      <output_dir>/<metric>/<sc_race>__<game>.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- robust JSON value coercion ----------------------

_NUM_KEYS = ("value", "val", "mean", "avg", "y", "item")

def _to_float(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return np.nan
    if isinstance(v, dict):
        # common numeric-like keys
        for k in _NUM_KEYS:
            if k in v:
                try:
                    return float(v[k])
                except Exception:
                    pass
        # fallback to first numeric-coercible value
        for x in v.values():
            try:
                return float(x)
            except Exception:
                continue
        return np.nan
    return np.nan


# value key aliases (case tolerant)
VAL_ALIASES: Dict[str, List[str]] = {
    "test_return_mean":       ["test_return_mean", "Test_return_mean"],
    "test_return_std":        ["test_return_std", "Test_return_std"],
    "test_battle_won_mean":   ["test_battle_won_mean", "Test_battle_won_mean"],
    "Training_avg_time":      ["Training_avg_time", "training_avg_time", "Training_avg_time_mean"],
}

def _step_keys(val_key: str) -> List[str]:
    return [
        f"{val_key}_T",          # preferred
        f"{val_key}_steps",
        f"{val_key}_x",
        "t_env_steps",
        "steps",
        "episode_T",
        "episodes",
    ]

def _first_present(d: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None

def _resolve_value_key(info: dict, base: str) -> Optional[str]:
    for k in VAL_ALIASES.get(base, [base]):
        if k in info:
            return k
    return None

def _extract_xy(info: dict, base: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns (x, y) arrays for metric 'base', or None if not found.
    Handles lists of numbers or lists of dicts.
    Drops NaNs and aligns lengths to min(len(x), len(y)).
    """
    vkey = _resolve_value_key(info, base)
    if vkey is None:
        return None

    raw_vals = info[vkey]
    if isinstance(raw_vals, list):
        y = np.asarray([_to_float(e) for e in raw_vals], dtype=float)
    else:
        y = np.asarray([_to_float(raw_vals)], dtype=float)

    skey = _first_present(info, _step_keys(vkey))
    if skey is not None:
        raw_steps = info[skey]
        if isinstance(raw_steps, list):
            x = np.asarray([_to_float(e) for e in raw_steps], dtype=float)
        else:
            x = np.asarray([_to_float(raw_steps)], dtype=float)
    else:
        x = np.arange(len(y), dtype=float)

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    # drop NaNs & keep alignment
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) == 0 or len(y) == 0:
        return None
    return x, y


# ---------------------- directory traversal & loading ----------------------

def _iter_scgame(root: Path) -> List[Tuple[str, str]]:
    """
    Yield (<sc_race>, <game>) for every ROOT/<sc_race>/<game>/ pair.
    """
    pairs = []
    for sc_race_dir in sorted(root.iterdir()):
        if not sc_race_dir.is_dir():
            continue
        for game_dir in sorted(sc_race_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            pairs.append((sc_race_dir.name, game_dir.name))
    return pairs

def _list_model_runs(root: Path, sc_race: str, game: str, model: str) -> List[Path]:
    model_dir = root / sc_race / game / model
    if not model_dir.is_dir():
        return []
    runs = []
    for run_dir in sorted(model_dir.iterdir()):
        if (run_dir / "info.json").is_file():
            runs.append(run_dir / "info.json")
    return runs

def _load_info(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None


# ---------------------- best-run selection per model ----------------------

def _best_run_for_model(root: Path, sc_race: str, game: str, model: str, verbose: bool=False) -> Optional[Path]:
    """
    Choose the run whose FINAL test_battle_won_mean is highest.
    Returns path to that run's info.json or None.
    """
    candidates = _list_model_runs(root, sc_race, game, model)
    if not candidates:
        if verbose:
            print(f"  [INFO] No runs for model '{model}' in {sc_race}/{game}")
        return None

    best_path = None
    best_final = -np.inf

    for info_path in candidates:
        info = _load_info(info_path)
        if info is None:
            continue

        xy = _extract_xy(info, "test_battle_won_mean")
        if xy is None:
            if verbose:
                print(f"    [SKIP] {info_path} has no test_battle_won_mean")
            continue

        _, y = xy
        # use last finite value as "final"
        finite = np.isfinite(y)
        if not np.any(finite):
            if verbose:
                print(f"    [SKIP] {info_path} test_battle_won_mean all NaN")
            continue
        final_val = float(y[np.where(finite)[0][-1]])

        if final_val > best_final:
            best_final = final_val
            best_path = info_path

    if verbose and best_path is not None:
        print(f"  [BEST] {model}: {best_path.parent.name} (final win={best_final:.3f})")
    if verbose and best_path is None:
        print(f"  [INFO] No valid runs for model '{model}' in {sc_race}/{game}")

    return best_path


# ---------------------- plotting helpers ----------------------

def _common_max_step_across_models(info_map: Dict[str, dict], metric: str) -> Optional[float]:
    """
    Among the selected models' CHOSEN runs (info_map[model] = info dict),
    compute min(last_x) for the metric (to align ranges).
    """
    last_steps = []
    for _, info in info_map.items():
        xy = _extract_xy(info, metric)
        if xy is None:
            continue
        x, _ = xy
        if x.size > 0:
            last_steps.append(float(x[np.isfinite(x)][-1]))
    if not last_steps:
        return None
    return min(last_steps)

def _plot_metric(
    outdir: Path,
    sc_race: str,
    game: str,
    model_to_info: Dict[str, dict],
    metric: str,
    ylabel: str,
    ylimit: Optional[Tuple[float, float]] = None,
    std_metric: Optional[str] = None,  # only used for test_return_mean
):
    # Ensure base output directory exists even if nothing gets plotted
    (outdir / metric).mkdir(parents=True, exist_ok=True)

    common_max = _common_max_step_across_models(model_to_info, metric)
    if common_max is None:
        # still keep the folder so the user sees structure
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted_any = False

    for m in sorted(model_to_info.keys()):
        info = model_to_info[m]
        xy = _extract_xy(info, metric)
        if xy is None:
            continue
        x, y = xy
        mask = x <= common_max
        x, y = x[mask], y[mask]
        if x.size == 0:
            continue

        ax.plot(x, y, label=m, linewidth=2.0)
        plotted_any = True

        if std_metric is not None:
            xy_std = _extract_xy(info, std_metric)
            if xy_std is not None:
                xs, ys = xy_std
                s_mean = pd.Series(y, index=x)
                s_std  = pd.Series(ys, index=xs)
                s_mean = s_mean.loc[s_mean.index <= common_max]
                s_std  = s_std.loc[s_std.index <= common_max]
                joined = pd.concat([s_mean, s_std], axis=1, join="inner")
                joined.columns = ["mean", "std"]
                if not joined.empty:
                    ax.fill_between(joined.index.values,
                                    (joined["mean"] - joined["std"]).values,
                                    (joined["mean"] + joined["std"]).values,
                                    alpha=0.18)

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_title(f"{sc_race} / {game}")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(*ylimit)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)

    fname = f"{sc_race}__{game}.png"
    fig.savefig((outdir / metric) / fname, dpi=240, bbox_inches="tight")
    plt.close(fig)


# -------------------------------- CLI & main ---------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sacred_directory", type=str, required=True, help="Root results directory")
    p.add_argument("--models", nargs="+", required=True, help="Model folder names to include")
    p.add_argument("--output_dir", type=str, default="plots_three", help="Where to save PNGs")
    p.add_argument("--verbose", action="store_true", help="Print diagnostics")
    args = p.parse_args()

    root = Path(args.sacred_directory)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)  # <-- create top folder immediately

    pairs = _iter_scgame(root)
    if not pairs:
        print(f"[WARN] No <sc_race>/<game> pairs found under {root}")
        print(f"[DONE] Plots saved to: {outdir.resolve()}")
        return

    for sc_race, game in pairs:
        if args.verbose:
            print(f"[PAIR] {sc_race} / {game}")
        # Select best run per model for this pair
        model_to_info: Dict[str, dict] = {}
        for m in args.models:
            best_info_path = _best_run_for_model(root, sc_race, game, m, verbose=args.verbose)
            if best_info_path is None:
                continue
            info = _load_info(best_info_path)
            if info is not None:
                model_to_info[m] = info

        if not model_to_info:
            if args.verbose:
                print(f"  [INFO] No valid models with runs in {sc_race}/{game}")
            # still leave the outdir tree; move on
            continue

        # Plot the three requested metrics
        _plot_metric(outdir, sc_race, game, model_to_info,
                     metric="test_return_mean",
                     ylabel="Reward",
                     ylimit=None,
                     std_metric="test_return_std")

        _plot_metric(outdir, sc_race, game, model_to_info,
                     metric="test_battle_won_mean",
                     ylabel="Win rate",
                     ylimit=(0.0, 1.0),
                     std_metric=None)

        _plot_metric(outdir, sc_race, game, model_to_info,
                     metric="Training_avg_time",
                     ylabel="Training avg time (s/update)",
                     ylimit=None,
                     std_metric=None)

    print(f"[DONE] Plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
