#!/bin/bash

# Activate venv
base_model=qwen3_32b
small_model=qwen3_1.7b
dataset=amc23
sample_size=40


JSON_DIR="margin_outputs/${dataset}_${base_model}_${small_model}"

echo "Starting profile analysis on $JSON_DIR..."
python prob_margin_profile.py \
    --json_dir "$JSON_DIR" \
    --min_count 1 \
    --sample ${sample_size} \
    --save_path "margin_outputs/${dataset}_${base_model}_${small_model}/profile_stats_${sample_size}.json"
