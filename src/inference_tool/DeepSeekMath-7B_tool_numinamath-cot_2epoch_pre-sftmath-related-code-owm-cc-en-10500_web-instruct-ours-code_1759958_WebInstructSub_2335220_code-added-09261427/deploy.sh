__conda_setup="$('/usr/local/lib/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/lib/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/lib/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/lib/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate /mnt/cache/luzimu/rlhf_math/.env/vllmenv

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_IB_AR_THRESHOLD=0

export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

hostname -I # print the host ip

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo "Running on $CHUNKS GPUs: ${GPULIST[@]}"

checkpoint=$1

python -m vllm.entrypoints.api_server \
--model /mnt/cache/luzimu/mathllm-finetune/out/DeepSeekMath-7B_tool_numinamath-cot_2epoch_pre-sftmath-related-code-owm-cc-en-10500_web-instruct-ours-code_1759958_WebInstructSub_2335220_code-added-09261427/checkpoint-${checkpoint} \
--trust-remote-code \
--seed 42 \
--tensor-parallel-size 1 \
--host 0.0.0.0 \
--port 8001