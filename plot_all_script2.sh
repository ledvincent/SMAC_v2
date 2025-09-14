#!/usr/bin/env bash

python -m src.utils.create_plots \
  --sacred_directory results/sacred2 \
  --output_dir results/plots \
  --test_only \
  --max_step 350 \
  --models hpn_att_rnn hpn_qmix role_kl_qmix role_sep_qmix ss_qmix role_rnn