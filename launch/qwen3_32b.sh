vllm serve Qwen/Qwen3-32B  \
    --dtype auto \
    -tp 2 \
    --max_model_len 36000 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --port 30000 \
    --chat-template utils/chat_template/qwen3_modified.jinja
