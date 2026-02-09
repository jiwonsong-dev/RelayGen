vllm serve Qwen/Qwen3-1.7B \
    --dtype auto \
    -tp 1 \
    --max_model_len 36000 \
    --gpu-memory-utilization 0.8 \
    --enable-prefix-caching \
    --dtype bfloat16 \
    --port 30002 \
    --chat-template utils/chat_template/qwen3_modified.jinja
