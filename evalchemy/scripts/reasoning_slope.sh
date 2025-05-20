export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export HF_HOME=/mnt/nas/dongkeun/huggingface
export TRANSFORMERS_CACHE=/mnt/nas/dongkeun/huggingface


export DS_SIZE=1000

export DS_PATH=DKYoon/r1-triviaqa-slope
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks reasoning_pre_cot\
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=8192,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/reasoning_slope/$OUTPUT_SUBDIR \


export DS_PATH=DKYoon/r1-nonambigqa-slope
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks reasoning_pre_cot\
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=8192,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/reasoning_slope/$OUTPUT_SUBDIR \