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

tag=$1
start_idx=$2
interval=$3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

tmux new-session -d -s deploy_$start_idx "bash $DIR/deploy.sh"
sleep 5s
tmux new-session -d -s infer_$start_idx "python $DIR/inference.py --ch $tag --start_idx $start_idx --interval $interval"
tmux new-session -d -s infer_cot_$start_idx "python $DIR/inference_cot.py --ch cot_$tag --start_idx $start_idx --interval $interval"
sleep 1s
tmux ls