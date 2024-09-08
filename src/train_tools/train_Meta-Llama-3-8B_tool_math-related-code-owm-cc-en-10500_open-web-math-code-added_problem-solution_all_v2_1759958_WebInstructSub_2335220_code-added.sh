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

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

conda activate /mnt/cache/sharemath/finetune_mixtral/mixtralenv1
cd /mnt/cache/luzimu/mathllm-finetune/src/train_tools

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

export NCCL_IB_TIMEOUT=22   
export NCCL_IB_RETRY_CNT=13 
export NCCL_IB_AR_THRESHOLD=0

wandb login "a8fe59167f5543baf6168a0cf5d52773a1bd6bf8"

OMP_NUM_THREADS=1 torchrun --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nproc_per_node 4 /mnt/cache/luzimu/mathllm-finetune/src/train_tools/train.py \
--ddp_timeout 3600 \
--processor dialogue \
--model_cfg /mnt/cache/luzimu/math_pretrain/outs/Meta-Llama3-8B_math-related-code-owm-cc-en_cc-en_textbooks_synthetic_code_open-web-math-08211038/checkpoint-10500 \
--train_file /mnt/cache/luzimu/mathllm-finetune/data/train/open-web-math-code-added_problem-solution_all_v2_1765680/data/train/train.jsonl,/mnt/cache/k12_data/data_lzm/WebInstruct/WebInstructSub_2335219.jsonl \
--output_dir /mnt/cache/luzimu/mathllm-finetune/out/Meta-Llama-3-8B_tool_math-related-code-owm-cc-en-10500_open-web-math-code-added_problem-solution_all_v2_1759958_WebInstructSub_2335220-code-added \
--logging_dir /mnt/cache/luzimu/mathllm-finetune/out/Meta-Llama-3-8B_tool_math-related-code-owm-cc-en-10500_open-web-math-code-added_problem-solution_all_v2_1759958_WebInstructSub_2335220-code-added \
--remove_unused_columns False \
--dataloader_num_workers 32 \
--max_len 2048 \
--max_steps -1 \
--num_train_epochs 2 \
--save_steps 10000 \
--warmup_steps 50 \
--logging_steps 10 \
--learning_rate 1e-5 \
--lr_scheduler_type cosine \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--seed 3407 \
--deepspeed /mnt/cache/luzimu/mathllm-finetune/src/train_tools/config/deepspeed.json \
--bf16 \
--do_train \
--save_safetensors \
--gradient_checkpointing \
--report_to wandb \
--run_name Meta-Llama-3-8B_tool_math-related-code-owm-cc-en-10500_open-web-math-code-added_problem-solution_all_v2_1759958_WebInstructSub_2335220-code-added \
--save_total_limit 1 \