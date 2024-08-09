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

conda activate /mnt/cache/luzimu/mathllm-finetune/.env/handbookenv
cd /mnt/cache/luzimu/mathllm-finetune

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

export NCCL_IB_TIMEOUT=22   
export NCCL_IB_RETRY_CNT=13 
export NCCL_IB_AR_THRESHOLD=0

wandb login "a8fe59167f5543baf6168a0cf5d52773a1bd6bf8"

CONFIG=${1}
echo ${CONFIG}

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /mnt/cache/luzimu/mathllm-finetune/recipes/accelerate_configs/deepspeed_zero3_8gpu.yaml /mnt/cache/luzimu/mathllm-finetune/scripts/run_sft_tool.py ${CONFIG}