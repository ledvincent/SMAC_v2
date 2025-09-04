#!/usr/bin/env bash

python -m src.utils.create_plots \
  --sacred_directory results/sacred \
  --output_dir results/plots \
  --test_only \
  --max_step 200 \