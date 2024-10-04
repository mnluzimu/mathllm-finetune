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

conda activate /mnt/cache/luzimu/rlhf_math/.env/inferenv

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

dataset=$1
round=$2
index=$3
model_path="/mnt/cache/luzimu/mathllm-finetune/out/Qwen2_5-Math-7B_numinamath-cot_2epoch-10022154"

tmux kill-session -t loop_${dataset}_${index}
tmux kill-session -t ${index}

tmux new-session -d -s deploy_${dataset}_${index} "bash $DIR/deploy_vllm.sh $model_path"
tmux new-session -d -s loop_${dataset}_${index} "bash $DIR/infer_loop.sh $dataset $round $index"
tmux new-session -d -s watch_${dataset}_${index} "bash $DIR/watch.sh $index $dataset"
