#!/bin/bash

small_model_size=qwen3_1.7b
base_model_size=qwen3_32b

dataset=aime25
budget=32768

path=outputs/relaygen-$small_model_size-$base_model_size$dataset
log_path=logs/relaygen-$small_model_size-$base_model_size/$dataset

mkdir -p $path
mkdir -p $log_path

for repeat_id in {0..3}
do
    for problem_id in {0..29}
    do
        python -m inference.relaygen \
            --dataset_name $dataset \
            --base_model_size $base_model_size \
            --small_model_size $small_model_size \
            --answer_with_small_model \
            --problem_id $problem_id \
            --repeat_id $repeat_id \
            --budget $budget \
            --output_dir $path \
            --verbose
    done
done