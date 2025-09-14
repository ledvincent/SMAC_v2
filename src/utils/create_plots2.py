#!/usr/bin/env python3
"""
create_plots.py — simple, robust plotting for SMAC/SMACv2 logs.

Key changes vs your previous script:
- No dependency on `battle_won_std` / `test_battle_won_std` (they're not required).
- Optional smoothing (EMA or centered moving average) applied PER RUN before aggregation.
- Adds a new metric family: training average time (training_avg_time), handled like battle_won.
- "Best run" is still picked by last available test_return_mean.
- Gracefully handles missing splits (e.g., if a metric has only train or only test series).

Example:
  python create_plots.py --sacred_directory ./sacred \
    --smooth_winrate_ema 21 \
    --smooth_time 31

Outputs:
  plots/<metric_folder>/(best_models|avg_models|subplots)/...
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

# =============================================================================
#                              Pretty helpers
# =============================================================================
def make_game_key(game_name: str, agents: str) -> str:
    return f"{game_name}__{agents}"

def split_game_key(game_key: str):
    if "__" in game_key:
        g, a = game_key.split("__", 1)
    else:
        g, a = game_key, ""
    return g, a

def pretty_game_title(game_key: str) -> str:
    g, a = split_game_key(game_key)
    return f"{g} / {a}" if a else g

# =============================================================================
#                              Metric registry
# =============================================================================
# Each metric_base maps to (train_mean_key, test_mean_key, train_std_key, test_std_key, label, folder, ylim, smoothable)
_METRICS = {
    "return": (
        "return_mean", "test_return_mean",
        "return_std",  "test_return_std",
        "Reward", "return_mean", None, False
    ),
    "battle_won": (
        "battle_won_mean", "test_battle_won_mean",
        None, None,
        "Win rate", "battle_won", (0.0, 1.0), True
    ),
    "train_time": (
        "training_avg_time", "test_training_avg_time",  # test_* often absent; handled gracefully
        None, None,
        "Training avg time", "training_avg_time", None, True
    ),
}

def _metric_keys(metric_base: str):
    """Return keys tuple for the given metric_base."""
    if metric_base not in _METRICS:
        raise ValueError(f"Unknown metric_base '{metric_base}'. Valid: {list(_METRICS.keys())}")
    return _METRICS[metric_base][:4]

def _metric_meta(metric_base: str):
    """Return (label, folder, ylim, smoothable)."""
    if metric_base not in _METRICS:
        raise ValueError(f"Unknown metric_base '{metric_base}'. Valid: {list(_METRICS.keys())}")
    return _METRICS[metric_base][4:]

# =============================================================================
#                              Data parsing
# =============================================================================
def info_parser(sacred_directory: Path):
    games = set()
    models = set()
    game_groups = {}
    model_game = {}
    game_model = {}

    sacred_directory = Path(sacred_directory)
    for game_dir in sacred_directory.iterdir():  # game_name
        if not game_dir.is_dir():
            continue
        game_name = game_dir.name
        game_groups.setdefault(game_name, set())

        for agents_dir in game_dir.iterdir():   # game_agents (e.g., 5v5)
            if not agents_dir.is_dir():
                continue
            agents = agents_dir.name
            game_key = make_game_key(game_name, agents)

            games.add(game_key)
            game_groups[game_name].add(game_key)
            game_model.setdefault(game_key, set())

            for model_dir in agents_dir.iterdir():  # model_name
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                models.add(model_name)
                model_game.setdefault(model_name, set()).add(game_key)
                game_model[game_key].add(model_name)

    return games, game_groups, models, model_game, game_model

def _to_float(v):
    try:
        if isinstance(v, dict):
            if "value" in v:
                return float(v["value"])
            for k in ("item", "val"):
                if k in v:
                    return float(v[k])
        return float(v)
    except Exception:
        return float("nan")

def _extract_metric_series_from_info(info_dict, metric_name):
    """Return (steps, values) for a metric; tolerate missing steps arrays and compute default 0..n-1."""
    if metric_name not in info_dict:
        return None, None
    raw_vals = info_dict[metric_name]
    if isinstance(raw_vals, list):
        values = [_to_float(x if not isinstance(x, dict) else x.get("value", x)) for x in raw_vals]
    else:
        values = [_to_float(raw_vals)]

    n = len(values)
    steps_candidates = [
        metric_name + "_steps",
        metric_name + "_x",
        "steps",
        "t_env_steps",
    ]
    steps = None
    for key in steps_candidates:
        if key in info_dict:
            cand = info_dict[key]
            if isinstance(cand, list) and len(cand) == n:
                steps = cand
                break
    if steps is None:
        steps = list(range(n))
    return steps, values

def results_parser(sacred_directory, models=None, game_filter=None):
    sacred_directory = Path(sacred_directory)

    # Discover structure
    all_games, game_groups, all_models, model_game, game_model = info_parser(sacred_directory)

    # Filter models if user specified
    if models is None:
        models = sorted(list(all_models))
    else:
        models = [m for m in models if m in all_models]

    # Filter by a specific base game if requested
    if game_filter is not None:
        if game_filter in game_groups:
            game_names = set(game_groups[game_filter])
            game_groups = {game_filter: game_names}
        else:
            print(f"[WARN] game_filter='{game_filter}' not found; using all games.")
            game_names = all_games
    else:
        game_names = all_games

    print('Games (subgames):', sorted(list(game_names)))

    # We'll attempt to read these keys when present; missing keys are simply skipped.
    metric_names = [
        "return_mean", "return_std", "test_return_mean", "test_return_std",
        "battle_won_mean", "test_battle_won_mean",
        "training_avg_time", "test_training_avg_time",
    ]
    dataframes = {g: {m: pd.DataFrame() for m in metric_names} for g in game_names}

    # Walk the directory tree: game/game_agents/model/run_num/info.json
    for game_key in game_names:
        base_game, agents = split_game_key(game_key)
        for model in models:
            model_dir = sacred_directory / base_game / agents / model
            if not model_dir.is_dir():
                continue

            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir() or not run_dir.name.isdigit():
                    continue

                info_path = run_dir / "info.json"
                if not info_path.is_file():
                    print(f"{info_path} not found: SKIPPED")
                    continue

                try:
                    with open(info_path, "r") as f:
                        info = json.load(f)
                except Exception as e:
                    print(f"[ERR] Failed to load {info_path}: {e}")
                    continue

                for metric in metric_names:
                    steps, values = _extract_metric_series_from_info(info, metric)
                    if steps is None or values is None or len(values) == 0:
                        continue
                    col_name = f"{model}_{run_dir.name}"
                    s = pd.Series(data=values, index=steps, name=col_name)
                    if dataframes[game_key][metric].empty:
                        dataframes[game_key][metric] = s.to_frame()
                    else:
                        dataframes[game_key][metric] = pd.concat(
                            [dataframes[game_key][metric], s], axis=1
                        )

    # Pick best run per model by last available test_return_mean
    best_model = {g: {} for g in game_names}
    for game_key in game_names:
        df_test_mean = dataframes[game_key].get('test_return_mean', pd.DataFrame())
        if df_test_mean.empty:
            for m in models:
                best_model[game_key][m] = None
            continue
        for m in models:
            cols = [c for c in df_test_mean.columns if c.rsplit('_', 1)[0] == m]
            if not cols:
                best_model[game_key][m] = None
                continue
            best_val = float('-inf')
            best_run = None
            for c in cols:
                series = df_test_mean[c].dropna()
                if series.empty:
                    continue
                val = series.iloc[-1]
                if val > best_val:
                    best_val = val
                    best_run = c.rsplit('_', 1)[1]
            best_model[game_key][m] = best_run
    return dataframes, best_model, game_names, game_groups, model_game, game_model

# =============================================================================
#                         Smoothing utilities
# =============================================================================
def _smooth_df(df: pd.DataFrame, ma_window: int = 0, ewm_span: int = 0) -> pd.DataFrame:
    """
    Apply smoothing to each run/column independently, preserving the index (steps).
    Prefer EWM if provided, else centered moving average.
    """
    if df is None or df.empty:
        return df
    df = df.sort_index()
    if ewm_span and ewm_span > 1:
        return df.ewm(span=int(ewm_span), adjust=False).mean()
    if ma_window and ma_window > 1:
        return df.rolling(window=int(ma_window), min_periods=1, center=True).mean()
    return df

def _get_smooth_cfg(metric_base: str, args):
    """Return smoothing config dict for a metric_base, or None if disabled."""
    _, _, _, _, _, _, _, smoothable = _METRICS[metric_base] if metric_base in _METRICS else (None,)*8
    if not smoothable:
        return None
    if metric_base == "battle_won":
        if args.smooth_winrate_ema and args.smooth_winrate_ema > 1:
            return {"ewm_span": args.smooth_winrate_ema, "ma_window": 0}
        if args.smooth_winrate and args.smooth_winrate > 1:
            return {"ewm_span": 0, "ma_window": args.smooth_winrate}
    if metric_base == "train_time":
        if args.smooth_time_ema and args.smooth_time_ema > 1:
            return {"ewm_span": args.smooth_time_ema, "ma_window": 0}
        if args.smooth_time and args.smooth_time > 1:
            return {"ewm_span": 0, "ma_window": args.smooth_time}
    return None

# =============================================================================
#                         Plot preparation (metricized)
# =============================================================================
def get_model_variant_data(dataframes, game_name, model_variant, metric_base='return', smooth_cfg=None):
    train_mean_key, test_mean_key, _, _ = _metric_keys(metric_base)

    # Pull wide DataFrames for available splits; allow missing keys gracefully
    df_wide_train = dataframes[game_name].get(train_mean_key, pd.DataFrame()).copy() if train_mean_key else pd.DataFrame()
    df_wide_test  = dataframes[game_name].get(test_mean_key,  pd.DataFrame()).copy() if test_mean_key  else pd.DataFrame()

    # Filter to target model runs
    if not df_wide_train.empty:
        cols = [c for c in df_wide_train.columns if c.rsplit('_', 1)[0] == model_variant]
        df_wide_train = df_wide_train[cols]
    if not df_wide_test.empty:
        cols = [c for c in df_wide_test.columns if c.rsplit('_', 1)[0] == model_variant]
        df_wide_test = df_wide_test[cols]

    # Apply optional smoothing per run
    if smooth_cfg:
        df_wide_train = _smooth_df(df_wide_train, **smooth_cfg) if not df_wide_train.empty else df_wide_train
        df_wide_test  = _smooth_df(df_wide_test,  **smooth_cfg) if not df_wide_test.empty  else df_wide_test

    return df_wide_train, df_wide_test

def prepare_avg_data_for_plotting(dataframes, game_name, game_model, ci_z=1.15, max_step=None, metric_base='return', smooth_cfg=None):
    models_to_plot = game_model[game_name]
    avg_data = {}

    for model in models_to_plot:
        avg_data[model] = {}
        df_train, df_test = get_model_variant_data(dataframes, game_name, model, metric_base=metric_base, smooth_cfg=smooth_cfg)
        for df, split in zip([df_train, df_test], ['train', 'test']):
            if df.empty:
                steps = np.array([]); mean = lower = upper = np.array([])
            else:
                if max_step is not None:
                    df = df[df.index <= max_step]
                steps = df.index.to_numpy()
                mean  = df.mean(axis=1).to_numpy()
                std   = df.std(axis=1).to_numpy()  # across runs
                n_runs = max(1, df.shape[1])
                sem   = std / np.sqrt(n_runs)
                ci    = ci_z * sem
                lower = mean - ci
                upper = mean + ci

            avg_data[model][f'{split}_steps'] = steps
            avg_data[model][f'{split}_mean']  = mean
            avg_data[model][f'{split}_lower'] = lower
            avg_data[model][f'{split}_upper'] = upper
    return avg_data

def prepare_best_data_for_plotting(dataframes, game_name, best_model, max_step=None, metric_base='return', smooth_cfg=None):
    models_to_plot = {m: r for m, r in best_model[game_name].items() if r is not None}
    best_data = {}
    train_mean_key, test_mean_key, train_std_key, test_std_key = _metric_keys(metric_base)

    for model, run_num in models_to_plot.items():
        best_data[model] = {}
        for split, mean_key, std_key in [
            ('train_', train_mean_key, train_std_key),
            ('test_',  test_mean_key,  test_std_key),
        ]:
            if mean_key is None:
                steps = np.array([]); mean = lower = upper = np.array([])
                best_data[model][f'{split}steps'] = steps
                best_data[model][f'{split}mean']  = mean
                best_data[model][f'{split}lower'] = lower
                best_data[model][f'{split}upper'] = upper
                continue

            df_mean = dataframes[game_name].get(mean_key, pd.DataFrame()).copy()
            df_std  = dataframes[game_name].get(std_key,  pd.DataFrame()).copy() if std_key else pd.DataFrame()

            col = f"{model}_{run_num}"
            if df_mean.empty or col not in df_mean.columns:
                steps = np.array([]); mean = lower = upper = np.array([])
            else:
                df_mean = df_mean[[col]]
                if max_step is not None:
                    df_mean = df_mean[df_mean.index <= max_step]
                if smooth_cfg:
                    df_mean = _smooth_df(df_mean, **smooth_cfg)

                steps = df_mean.index.to_numpy()
                mean  = df_mean.iloc[:, 0].to_numpy()

                # If we actually have a std series for this metric/split, use it for a band; otherwise skip bands.
                if not df_std.empty and col in df_std.columns:
                    df_std = df_std[[col]]
                    if max_step is not None:
                        df_std = df_std[df_std.index <= max_step]
                    if smooth_cfg:
                        df_std = _smooth_df(df_std, **smooth_cfg)
                    std = df_std.iloc[:, 0].to_numpy()
                    lower = mean - std
                    upper = mean + std
                else:
                    lower = np.array([])
                    upper = np.array([])

            best_data[model][f'{split}steps'] = steps
            best_data[model][f'{split}mean']  = mean
            best_data[model][f'{split}lower'] = lower
            best_data[model][f'{split}upper'] = upper
    return best_data

# =============================================================================
#                               Plotting
# =============================================================================
def palette_choice(models, test_only):
    models = list(models)
    if test_only:
        palette = sns.color_palette("hls", n_colors=len(models))
        model_color = {m: [c] for m, c in zip(models, palette)}
    else:
        palette = sns.color_palette("Paired", n_colors=len(models) * 2)
        model_color = {}
        for i, m in enumerate(models):
            model_color[m] = [palette[2 * i], palette[2 * i + 1]]
    return model_color

def common_legend(all_models, palette, test_only):
    legend_handles = []
    for model in all_models:
        linestyle = '--' if model == model.split('_')[0] else '-'
        colors = palette[model]
        if test_only:
            handle = Line2D([0], [0], color=colors[0], linestyle=linestyle, label=model)
            legend_handles.append(handle)
        else:
            handle_train = Line2D([0], [0], color=colors[0], linestyle=linestyle, label=model+'_train')
            handle_test  = Line2D([0], [0], color=colors[1], linestyle=linestyle, label=model+'_test')
            legend_handles.extend([handle_train, handle_test])
    return legend_handles

def build_output(best_or_avg, is_subplot, output_dir, filename, test_only, metric_folder):
    root = Path(output_dir) if output_dir else Path('plots')
    root = root / metric_folder  # metric-specific subfolder
    test_or_train = 'test_only' if test_only else 'train_and_test'
    folder = (root / 'subplots' / f"{best_or_avg}_subplots" / test_or_train) if is_subplot else (root / f"{best_or_avg}_models" / test_or_train)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{filename}.png"

def _has_band(arr):
    return isinstance(arr, np.ndarray) and arr.size > 0

def create_plots(
    data, game_name, best_or_avg, output_dir, palette=None, test_only=False,
    fill_between=True, ax=None, metric_label="Reward", metric_folder="return_mean", ylim=None
):
    models_to_plot = list(data.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
        new_fig = True
    else:
        new_fig = False

    sns.set_theme(style="whitegrid", context="talk")
    linewidth = 1.2

    model_color = palette_choice(models_to_plot, test_only) if palette is None else palette

    for model in models_to_plot:
        linestyle = 'dotted' if model == model.split('_')[0] else 'solid'
        if not test_only:
            to_plot = ['train', 'test']; labels = [model + '_train', model + '_test']
        else:
            to_plot = ['test']; labels = [model]

        for i, (typee, label) in enumerate(zip(to_plot, labels)):
            steps_key = f"{typee}_steps"
            mean_key  = f"{typee}_mean"
            low_key   = f"{typee}_lower"
            up_key    = f"{typee}_upper"
            if steps_key not in data[model] or mean_key not in data[model]:
                continue

            xdata = data[model][steps_key]
            ydata = data[model][mean_key]
            if xdata.size == 0 or ydata.size == 0:
                continue

            col = model_color[model][i]
            plt_kwargs = dict(label=label, color=col, linestyle=linestyle, linewidth=linewidth, legend=False, ax=ax)
            sns.lineplot(x=xdata, y=ydata, **plt_kwargs)

            lower = data[model].get(low_key, None)
            upper = data[model].get(up_key, None)
            if fill_between and isinstance(lower, np.ndarray) and isinstance(upper, np.ndarray) and lower.size and upper.size:
                ax.fill_between(xdata, lower, upper, color=col, alpha=0.15)

    title = pretty_game_title(game_name)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if new_fig:
        legend_handles = common_legend(models_to_plot, model_color, test_only)
        fig.legend(handles=legend_handles, loc='lower center', ncols=max(1, len(models_to_plot)), frameon=False,
                   fontsize='small', handletextpad=0.5, columnspacing=1, bbox_to_anchor=(0.5, 0.0))
        ax.set_xlabel('Training steps')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{best_or_avg} model {metric_label.lower()} evolution for {title}')
        filename = f"{best_or_avg}_model_{game_name}" + ("_test_only" if test_only else "")
        out_path = build_output(best_or_avg, False, output_dir, filename, test_only, metric_folder)
        plt.savefig(out_path, dpi=400)
        plt.close()
    else:
        ax.set_title(title)

def create_subplots(
    base_game, same_game, dataframes, all_models, model_type, best_or_avg, output_dir,
    test_only=False, fill_between=True, max_step=None, metric_base='return',
    metric_label="Reward", metric_folder="return_mean", ylim=None, smooth_cfg=None
):
    n_tasks = len(same_game)
    fig, axs = plt.subplots(1, n_tasks, figsize=(15 * n_tasks, 15), sharex=True)
    if n_tasks == 1:
        axs = [axs]

    fig.suptitle(f"{best_or_avg} model {metric_label.lower()} evolution for tasks of {base_game}")

    plot_function = prepare_best_data_for_plotting if best_or_avg == 'best' else prepare_avg_data_for_plotting
    palette = palette_choice(all_models, test_only)
    legend_handles = common_legend(all_models, palette, test_only)

    for ax, game_name in zip(axs, same_game):
        plot_data = plot_function(
            dataframes, game_name, model_type, max_step=max_step, metric_base=metric_base, smooth_cfg=smooth_cfg
        )
        create_plots(
            plot_data, game_name, best_or_avg, output_dir=None, palette=palette,
            test_only=test_only, fill_between=fill_between, ax=ax,
            metric_label=metric_label, metric_folder=metric_folder, ylim=ylim
        )

    mid = len(axs) // 2
    axs[mid].set_xlabel("Training steps", labelpad=20)
    axs[0].set_ylabel(metric_label, labelpad=20)
    if ylim is not None:
        for ax in axs:
            ax.set_ylim(*ylim)

    fig.legend(handles=legend_handles, loc='lower center', ncols=max(1, len(all_models)), frameon=False,
               fontsize='medium', handletextpad=0.5, columnspacing=1, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    filename = f"{best_or_avg}_model_{base_game}_subtasks" + ("_test_only" if test_only else "")
    out_path = build_output(best_or_avg, True, output_dir, filename, test_only, metric_folder)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)

# =============================================================================
#                                  MAIN
# =============================================================================
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--sacred_directory", type=str, default="./sacred")
    p.add_argument("--models", nargs='+', default=None)
    p.add_argument("--game", type=str, required=False)
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--no_fill_between", action="store_false")
    p.add_argument("--linestyle", type=str, default='solid')
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--output_file_name", type=str, default=None)
    p.add_argument("--normalized", action="store_true")
    p.add_argument("--max_step", type=int, default=None, help="Maximum step to plot (crops x-axis)")
    # --- Smoothing options (applied per run BEFORE aggregation) ---
    p.add_argument("--smooth_winrate", type=int, default=0, help="Centered moving average window for battle_won (0=off).")
    p.add_argument("--smooth_winrate_ema", type=int, default=0, help="EMA span for battle_won (0=off; overrides --smooth_winrate).")
    p.add_argument("--smooth_time", type=int, default=0, help="Centered moving average window for training_avg_time (0=off).")
    p.add_argument("--smooth_time_ema", type=int, default=0, help="EMA span for training_avg_time (0=off; overrides --smooth_time).")
    return p.parse_args()

def run_metric_suite(metric_base: str, dataframes, best_model, game_names, game_groups, game_model, args):
    label, folder, ylim, _ = _metric_meta(metric_base)
    smooth_cfg = _get_smooth_cfg(metric_base, args)

    print(f'Plotting standalone subgames ({metric_base})')
    for game_name in sorted(list(game_names)):
        # Best
        best_data = prepare_best_data_for_plotting(
            dataframes, game_name, best_model, max_step=args.max_step, metric_base=metric_base, smooth_cfg=smooth_cfg
        )
        create_plots(
            best_data, game_name, 'best', args.output_dir, None, args.test_only,
            args.no_fill_between, metric_label=label, metric_folder=folder, ylim=ylim
        )
        # Average
        avg_data = prepare_avg_data_for_plotting(
            dataframes, game_name, game_model, max_step=args.max_step, metric_base=metric_base, smooth_cfg=smooth_cfg
        )
        create_plots(
            avg_data, game_name, 'avg', args.output_dir, None, args.test_only,
            args.no_fill_between, metric_label=label, metric_folder=folder, ylim=ylim
        )

    # Subplots per base game
    base_game_model = {}
    for base_game, subgames in game_groups.items():
        base_game_model[base_game] = set()
        for subgame in subgames:
            base_game_model[base_game].update(list(game_model.get(subgame, [])))

    print(f'Plotting subplots per base game ({metric_base})')
    for base_game, subgames in game_groups.items():
        same_game = sorted(list(subgames))
        all_models = sorted(list(base_game_model[base_game]))
        # Best
        create_subplots(
            base_game, same_game, dataframes, all_models, best_model, 'best', args.output_dir,
            args.test_only, args.no_fill_between, max_step=args.max_step,
            metric_base=metric_base, metric_label=label, metric_folder=folder, ylim=ylim, smooth_cfg=smooth_cfg
        )
        # Avg
        create_subplots(
            base_game, same_game, dataframes, all_models, game_model, 'avg', args.output_dir,
            args.test_only, args.no_fill_between, max_step=args.max_step,
            metric_base=metric_base, metric_label=label, metric_folder=folder, ylim=ylim, smooth_cfg=smooth_cfg
        )

if __name__ == "__main__":
    args = cli()
    dataframes, best_model, game_names, game_groups, model_game, game_model = results_parser(
        args.sacred_directory, args.models, args.game
    )

    # 1) Rewards -> plots/return_mean/...
    run_metric_suite(metric_base="return", dataframes=dataframes, best_model=best_model,
                     game_names=game_names, game_groups=game_groups, game_model=game_model, args=args)

    # 2) Win rate -> plots/battle_won/... (y in [0,1])
    run_metric_suite(metric_base="battle_won", dataframes=dataframes, best_model=best_model,
                     game_names=game_names, game_groups=game_groups, game_model=game_model, args=args)

    # 3) Training avg time -> plots/training_avg_time/...
    run_metric_suite(metric_base="train_time", dataframes=dataframes, best_model=best_model,
                     game_names=game_names, game_groups=game_groups, game_model=game_model, args=args)
