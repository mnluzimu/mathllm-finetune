tag=$1
start_idx=$2
interval=$3
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd /

conda init bash
apt-get update
apt-get install tmux -y
source ~/.bashrc

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/cache/luzimu/rlhf_math/.env/inferenv

tmux kill-server
tmux new-session -d -s $tag "python $DIR/inference.py --ch $tag --start_idx $start_idx --interval $interval"
sleep 1s
tmux ls