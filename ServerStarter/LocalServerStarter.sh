#!/usr/bin/env bash
numservers=3
for i in `seq 0 $((numservers-1))`;
do
python server.py \
     --servers=$numservers \
     --task_index=$i &
sleep 2s
done