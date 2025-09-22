#!/usr/bin/env bash

python -m src.utils.create_plots2 \
  --sacred_directory results/sacred2 \
  --output_dir results/plots \
  --test_only \
  --max_step 350 \
  --smooth_winrate_ema 3 \
  --models hpn_qmix role_qmix role_sep_qmix ss_qmix role_rnn