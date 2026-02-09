vllm serve RedHatAI/Qwen3-32B-speculator.eagle3 \
    --dtype auto \
    -tp 2 \
    --max_model_len 36000 \
    --gpu-memory-utilization 0.8 \
    --enable-prefix-caching \
    --dtype bfloat16 \
    --port 30001 \
    --chat-template utils/chat_template/qwen3_modified.jinja
