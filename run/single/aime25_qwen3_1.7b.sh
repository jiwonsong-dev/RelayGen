model_size=qwen3_1.7b
dataset=aime25
path=outputs/$model_size/$dataset
mkdir $path
for repeat_id in {0..3}
do
    for problem_id in {0..29}
    do
        python -m inference.single \
                    --dataset_name $dataset \
                    --model_size $model_size \
                    --problem_id $problem_id \
                    --repeat_id $repeat_id \
                    --exit_period $exitperiod \
                    --budget 32768 \
                    --output_dir $path
    done
done
