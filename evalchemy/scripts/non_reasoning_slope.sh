export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export DS_SIZE=1000

export DS_PATH=DKYoon/qwen25-triviaqa-slope
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks non_reasoning_pre_cot\
    --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=8192,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/non_reasoning_slope/$OUTPUT_SUBDIR \


export DS_PATH=DKYoon/qwen25-nonambigqa-slope
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks non_reasoning_pre_cot\
    --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=8192,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/non_reasoning_slope/$OUTPUT_SUBDIR \
