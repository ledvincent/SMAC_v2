#!/usr/bin/env bash

python -m src.utils.create_plots3 \
  --sacred_directory ./results/sacred \
  --models hpn_qmix role_qmix role_sep_qmix ss_qmix \
  --output_dir plots_three \
  --verbose
