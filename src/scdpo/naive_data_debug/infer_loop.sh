dataset=$1
round=$2
index=$3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

while true; do
    tmux new-session -d -s $index "python $DIR/lce_solution_gen.py -r $round -i $index -d $dataset"
    sleep 5
    
    while true; do
        sleep 5
        python $DIR/process.py -r $round -i $index -d $dataset && break
        sleep 30s
    done

    ((round++))

    sleep 5
done
