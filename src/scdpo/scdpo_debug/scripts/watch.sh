index=$1
dataset=$2


while true; do
    python /mnt/cache/luzimu/rlhf_math/src/step_controled_dpo_lce_internlm_no_ch/watch_and_restart_${dataset}.py ${index}

    sleep 5
done