dataset=amc23
base_model=qwen3_32b
small_model=qwen3_1.7b
small_model_path=Qwen/Qwen3-1.7B
num_examples=160

python prob_margin_analysis.py \
  --model_path $small_model_path \
  --result_dir outputs/profile-${base_model}/${dataset} \
  --dataset_name $dataset \
  --num_examples $num_examples \
  --save_dir margin_outputs/${dataset}_${base_model}_${small_model} \
  --tensor_parallel_size 1 \
  --max_model_len 36000 \
  --gpu_memory_utilization 0.8 \
  --cuda 0 \
  --cue_effects \
  --min_occurrences 1 \
  --min_delta 0.0
