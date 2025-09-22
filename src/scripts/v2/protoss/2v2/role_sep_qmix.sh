#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# From src/scripts/v2/protoss/5v5 -> repo root is five levels up
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"

cd "$REPO_ROOT"

# If your SC2 install is under the SMAC repo, point PySC2 to it:
# export SC2PATH="$REPO_ROOT/3rdparty/StarCraftII"

# Params
GPU_ID=0
CONFIG=role_sep_qmix
ENV_CONFIG=sc2_v2_protoss
SEEDS=(0)
max_steps=1500000

N_UNITS=2
N_ENEMIES=2
TEAM_WEIGHTS='[0.5,0.5,0.0]'

OBS_AGENT_ID=False
OBS_LAST_ACTION=False
USE_WANDB=False
name="role_sep_qmix"
role_diversity=False
reward_win=200

for SEED in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" python src/main.py \
    --config="$CONFIG" \
    --env-config="$ENV_CONFIG" \
    with \
      env_args.capability_config.n_units="$N_UNITS" \
      env_args.capability_config.n_enemies="$N_ENEMIES" \
      env_args.capability_config.team_gen.weights="$TEAM_WEIGHTS" \
      t_max="$max_steps" \
      obs_agent_id="$OBS_AGENT_ID" \
      name="$name" \
      obs_last_action="$OBS_LAST_ACTION" \
      env_args.obs_last_action="$OBS_LAST_ACTION" \
      env_args.seed="$SEED" \
      env_args.reward_win="$reward_win" \
      role_diversity="$role_diversity" \
      use_wandb="$USE_WANDB"
done