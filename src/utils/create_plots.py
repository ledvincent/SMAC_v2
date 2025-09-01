#!/usr/bin/env python3
import argparse
import json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import re


# New Sacred layout:
# sacred/
#   game_name/
#     game_agents/            # e.g., 5v5, 10v10
#       model_name/
#         run_num/            # e.g., 0, 1, 2...
#           metrics.json


# ----------------------------------------------------------------------
#                       Helpers for naming/pretty titles
# ----------------------------------------------------------------------
def make_game_key(game_name: str, agents: str) -> str:
    """Internal key for a subgame: 'game__agents'."""
    return f"{game_name}__{agents}"

def split_game_key(game_key: str):
    """Reverse of make_game_key."""
    if "__" in game_key:
        g, a = game_key.split("__", 1)
    else:
        # fallback: treat entire key as game
        g, a = game_key, ""
    return g, a

def pretty_game_title(game_key: str) -> str:
    g, a = split_game_key(game_key)
    return f"{g} / {a}" if a else g


# ----------------------------------------------------------------------
#                        Data parsing (new layout)
# ----------------------------------------------------------------------
def info_parser(sacred_directory: Path):
    """
    Scans sacred_directory structured as:
      sacred/game_name/game_agents/model_name/run_num
    Returns:
      games:          set[str]            # subgame keys 'game__agents'
      game_groups:    dict[str, set[str]] # base game -> set of subgame keys
      models:         set[str]            # all model names found
      model_game:     dict[str, set[str]] # model -> set of subgame keys it appears in
      game_model:     dict[str, set[str]] # subgame key -> set of models found
    """
    games = set()
    models = set()
    game_groups = {}   # {game_name: set(subgame_keys)}
    model_game = {}    # {model_name: set(subgame_keys)}
    game_model = {}    # {subgame_key: set(model_names)}

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
    """Best-effort conversion to float for values coming from Sacred/NumPy dumps."""
    try:
        if isinstance(v, dict):
            # common sacred pattern: {"py/object": "numpy.float64", "dtype":"...", "value": X}
            if "value" in v:
                return float(v["value"])
            # fallback: some dumps put scalar directly under other keys
            for k in ("item", "val"):
                if k in v:
                    return float(v[k])
        return float(v)
    except Exception:
        return float("nan")


def _extract_metric_series_from_info(info_dict, metric_name):
    """
    Return (steps, values) for a given metric from info.json.

    Supported layouts:
      metric_name : [ {"value": v1}, {"value": v2}, ... ]
      metric_name : [ v1, v2, ... ]
    Optional steps (any one of):
      metric_name + "_steps"
      metric_name + "_x"
      "steps"
      "t_env_steps"  # if provided as a list
    If none found, steps = range(len(values)).
    """
    if metric_name not in info_dict:
        return None, None

    raw_vals = info_dict[metric_name]
    # Normalize to a list of floats
    if isinstance(raw_vals, list):
        values = [_to_float(x if not isinstance(x, dict) else x.get("value", x)) for x in raw_vals]
    else:
        # unexpected shape; try to coerce single scalar
        values = [_to_float(raw_vals)]

    n = len(values)

    # Try to find a steps array
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
    """
    Builds DataFrames for each subgame (game__agents) and metric,
    reading from info.json in: sacred/game/agents/model/run_num/info.json.

    Also picks best run per model by last test_return_mean.
    """
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

    metric_names = ['return_mean', 'return_std', 'test_return_mean', 'test_return_std']
    dataframes = {g: {m: pd.DataFrame() for m in metric_names} for g in game_names}

    # Walk the new directory tree: game/game_agents/model/run_num/info.json
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

                # For each metric, add a Series column
                for metric in metric_names:
                    steps, values = _extract_metric_series_from_info(info, metric)
                    if steps is None or values is None or len(values) == 0:
                        # It's fine if some metrics are missing (e.g., only test_* available)
                        # print(f"[WARN] No values for {metric} in {info_path}")
                        continue

                    col_name = f"{model}_{run_dir.name}"
                    s = pd.Series(data=values, index=steps, name=col_name)

                    if dataframes[game_key][metric].empty:
                        dataframes[game_key][metric] = s.to_frame()
                    else:
                        # Align on index when concatenating (outer join on steps)
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
            # columns for this model: "<model>_<run>"
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
                val = series.iloc[-1]  # last recorded test_return_mean
                if val > best_val:
                    best_val = val
                    best_run = c.rsplit('_', 1)[1]  # run number as string
            best_model[game_key][m] = best_run

    return dataframes, best_model, game_names, game_groups, model_game, game_model



# ----------------------------------------------------------------------
#                         Plot preparation (unchanged)
# ----------------------------------------------------------------------
def get_model_variant_data(dataframes, game_name, model_variant):
    df_wide_train = dataframes[game_name]['return_mean'].copy()
    df_wide_test  = dataframes[game_name]['test_return_mean'].copy()
    # allow missing one of them
    common_cols = sorted(list(set(df_wide_train.columns) & set(df_wide_test.columns)))
    df_wide_train = df_wide_train[common_cols]
    df_wide_test  = df_wide_test[common_cols]

    models_to_plot = [c for c in common_cols if c.rsplit('_', 1)[0] == model_variant]
    df_wide_train = df_wide_train[models_to_plot]
    df_wide_test  = df_wide_test[models_to_plot]
    return df_wide_train, df_wide_test


def prepare_avg_data_for_plotting(dataframes, game_name, game_model, ci_z=1.15):
    # ci_z=1.15 ~ 75% CI; using columns (#runs) for SEM
    models_to_plot = game_model[game_name]
    avg_data = {}

    for model in models_to_plot:
        avg_data[model] = {}
        df_train, df_test = get_model_variant_data(dataframes, game_name, model)
        for df, split in zip([df_train, df_test], ['train', 'test']):
            if df.empty:
                steps = np.array([])
                mean  = np.array([])
                lower = upper = np.array([])
            else:
                steps = df.index.to_numpy()
                mean  = df.mean(axis=1).to_numpy()
                std   = df.std(axis=1).to_numpy()
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


def prepare_best_data_for_plotting(dataframes, game_name, best_model):
    models_to_plot = {m: r for m, r in best_model[game_name].items() if r is not None}
    best_data = {}

    for model, run_num in models_to_plot.items():
        best_data[model] = {}
        for prefix in ['', 'test_']:
            df_mean = dataframes[game_name].get(prefix + 'return_mean', pd.DataFrame())
            df_std  = dataframes[game_name].get(prefix + 'return_std',  pd.DataFrame())
            split   = 'train_' if prefix == '' else 'test_'

            if df_mean.empty or f"{model}_{run_num}" not in df_mean.columns:
                steps = np.array([]); mean = lower = upper = np.array([])
            else:
                steps = df_mean.index.to_numpy()
                mean  = df_mean[f"{model}_{run_num}"].to_numpy()
                std   = df_std[f"{model}_{run_num}"].to_numpy() if not df_std.empty and f"{model}_{run_num}" in df_std.columns else np.zeros_like(mean)
                lower = mean - std
                upper = mean + std

            best_data[model][f'{split}steps'] = steps
            best_data[model][f'{split}mean']  = mean
            best_data[model][f'{split}lower'] = lower
            best_data[model][f'{split}upper'] = upper
    return best_data


# ----------------------------------------------------------------------
#                              Plotting
# ----------------------------------------------------------------------
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

def build_output(best_or_avg, is_subplot, output_dir, filename, test_only):
    root = Path(output_dir) if output_dir else Path('plots')
    test_or_train = 'test_only' if test_only else 'train_and_test'
    folder = (root / 'subplots' / f"{best_or_avg}_subplots" / test_or_train) if is_subplot else (root / f"{best_or_avg}_models" / test_or_train)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{filename}.png"

def create_plots(data, game_name, best_or_avg, output_dir, palette=None, test_only=False, fill_between=True, ax=None):
    models_to_plot = list(data.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 15))
        new_fig = True
    else:
        new_fig = False

    sns.set_theme(style="whitegrid", context="talk")
    linewidth = 10 / max(1, len(models_to_plot))

    model_color = palette_choice(models_to_plot, test_only) if palette is None else palette

    for model in models_to_plot:
        linestyle = 'dotted' if model == model.split('_')[0] else 'solid'
        if not test_only:
            to_plot = ['train', 'test']; labels = [model + '_train', model + '_test']
        else:
            to_plot = ['test']; labels = [model]

        for i, (typee, label) in enumerate(zip(to_plot, labels)):
            if f"{typee}_steps" not in data[model]:
                continue
            col   = model_color[model][i]
            xdata = data[model][typee+'_steps']
            ydata = data[model][typee+'_mean']
            plt_kwargs = dict(label=label, color=col, linestyle=linestyle, linewidth=linewidth, legend=False, ax=ax)
            sns.lineplot(x=xdata, y=ydata, **plt_kwargs)

            if fill_between and f"{typee}_lower" in data[model] and f"{typee}_upper" in data[model]:
                lower = data[model][typee+'_lower']
                upper = data[model][typee+'_upper']
                ax.fill_between(xdata, lower, upper, color=col, alpha=0.15)

    title = pretty_game_title(game_name)
    if new_fig:
        legend_handles = common_legend(models_to_plot, model_color, test_only)
        fig.legend(handles=legend_handles, loc='lower center', ncols=max(1, len(models_to_plot)), frameon=False,
                   fontsize='small', handletextpad=0.5, columnspacing=1, bbox_to_anchor=(0.5, 0.0))
        ax.set_xlabel('Training steps')
        ax.set_ylabel('Reward')
        ax.set_title(f'{best_or_avg} model reward evolution for {title}')
        filename = f"{best_or_avg}_model_{game_name}" + ("_test_only" if test_only else "")
        out_path = build_output(best_or_avg, False, output_dir, filename, test_only)
        plt.savefig(out_path, dpi=400)
        plt.close()
    else:
        ax.set_title(title)

def create_subplots(base_game, same_game, dataframes, all_models, model_type, best_or_avg, output_dir, test_only=False, fill_between=True):
    n_tasks = len(same_game)
    fig, axs = plt.subplots(1, n_tasks, figsize=(15 * n_tasks, 15), sharex=True)
    if n_tasks == 1:
        axs = [axs]

    fig.suptitle(f"{best_or_avg} model reward evolution for tasks of {base_game}")

    plot_function = prepare_best_data_for_plotting if best_or_avg == 'best' else prepare_avg_data_for_plotting
    palette = palette_choice(all_models, test_only)
    legend_handles = common_legend(all_models, palette, test_only)

    for ax, game_name in zip(axs, same_game):
        plot_data = plot_function(dataframes, game_name, model_type)
        create_plots(plot_data, game_name, best_or_avg, output_dir=None, palette=palette, test_only=test_only, fill_between=fill_between, ax=ax)

    mid = len(axs) // 2
    axs[mid].set_xlabel("Training steps", labelpad=20)
    axs[0].set_ylabel('Episodic Reward', labelpad=20)

    fig.legend(handles=legend_handles, loc='lower center', ncols=max(1, len(all_models)), frameon=False,
               fontsize='medium', handletextpad=0.5, columnspacing=1, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    filename = f"{best_or_avg}_model_{base_game}_subtasks" + ("_test_only" if test_only else "")
    out_path = build_output(best_or_avg, True, output_dir, filename, test_only)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)


# ----------------------------------------------------------------------
#                                  MAIN
# ----------------------------------------------------------------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--sacred_directory", type=str, default="./sacred")  # root of the new layout
    p.add_argument("--models", nargs='+', default=None)                  # restrict to specific model names
    p.add_argument("--game", type=str, required=False)                  # filter to one base game_name
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--no_fill_between", action="store_false")
    p.add_argument("--linestyle", type=str, default='solid')
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--output_file_name", type=str, default=None)
    p.add_argument("--normalized", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = cli()

    dataframes, best_model, game_names, game_groups, model_game, game_model = results_parser(
        args.sacred_directory, args.models, args.game
    )

    print('Plotting standalone subgames')
    for game_name in sorted(list(game_names)):
        # Best
        best_data = prepare_best_data_for_plotting(dataframes, game_name, best_model)
        create_plots(best_data, game_name, 'best', args.output_dir, None, args.test_only, args.no_fill_between)
        # Average
        avg_data = prepare_avg_data_for_plotting(dataframes, game_name, game_model)
        create_plots(avg_data, game_name, 'avg', args.output_dir, None, args.test_only, args.no_fill_between)

    # Models per base game (for subplots)
    base_game_model = {}
    for base_game, subgames in game_groups.items():
        base_game_model[base_game] = set()
        for subgame in subgames:
            base_game_model[base_game].update(list(game_model.get(subgame, [])))

    print('Plotting subplots per base game')
    for base_game, subgames in game_groups.items():
        same_game = sorted(list(subgames))
        all_models = sorted(list(base_game_model[base_game]))
        # Best
        create_subplots(base_game, same_game, dataframes, all_models, best_model, 'best', args.output_dir, args.test_only, args.no_fill_between)
        # Avg
        create_subplots(base_game, same_game, dataframes, all_models, game_model, 'avg', args.output_dir, args.test_only, args.no_fill_between)
