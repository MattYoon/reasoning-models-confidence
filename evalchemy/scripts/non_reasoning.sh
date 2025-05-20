export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export HF_HOME=/mnt/nas/dongkeun/huggingface
export TRANSFORMERS_CACHE=/mnt/nas/dongkeun/huggingface


export DS_SIZE=1000

export MAX_LENGTH=4096
export DS_PATH=DKYoon/triviaqa_val_1k
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks non_reasoning\
    --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=8192,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/non_reasoning/$OUTPUT_SUBDIR \


export MAX_LENGTH=4096
export DS_PATH=DKYoon/nonambigqa_val_1k
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks non_reasoning\
    --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=8192,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/non_reasoning/$OUTPUT_SUBDIR \


# longer length for reasoning-intensive tasks
export MAX_LENGTH=8192
export DS_PATH=DKYoon/mmlupro_calc_val_1k
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks non_reasoning\
    --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=12288,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/non_reasoning/$OUTPUT_SUBDIR \


export MAX_LENGTH=8192
export DS_PATH=DKYoon/mmlupro_non_calc_val_1k
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks non_reasoning\
    --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=12288,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/non_reasoning/$OUTPUT_SUBDIR \


export MAX_LENGTH=8192
export DS_PATH=DKYoon/supergpqa_calc_middle_val_1k
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks non_reasoning\
    --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=12288,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/non_reasoning/$OUTPUT_SUBDIR \


export MAX_LENGTH=8192
export DS_PATH=DKYoon/supergpqa_non_calc_middle_val_1k
echo "Running dataset: $DS_PATH, Size: $DS_SIZE"
OUTPUT_SUBDIR=${DS_PATH#*/}

python -m eval.eval \
    --model vllm \
    --tasks non_reasoning\
    --model_args pretrained=Qwen/Qwen2.5-32B-Instruct,tensor_parallel_size=2,dtype="bfloat16",enforce_eager=True,gpu_memory_utilization=0.90,max_model_len=12288,data_parallel_size=1 \
    --batch_size auto \
    --output_path logs/non_reasoning/$OUTPUT_SUBDIR \