#!/usr/bin/env bash
numservers=3
for i in `seq 0 $numservers-1`;
do
python3 server.py \
     --servers=$numservers \
     --task_index=$i &
sleep 5s
done