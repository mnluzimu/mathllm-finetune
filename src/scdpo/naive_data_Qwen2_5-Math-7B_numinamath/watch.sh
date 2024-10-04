index=$1
dataset=$2

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

while true; do
    python $DIR/watch_and_restart.py -i ${index} -d ${dataset}
    sleep 5
done