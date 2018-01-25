#!/usr/bin/env bash

#rm nohup.out
workers=3
ps=1
for ((i=0; i<$ps; i++)); do
   python trainer_DILAB_local.py \
      ps \
      $i \
      --worker_num $workers \
      --ps_num $ps \
      &
done


for ((i=0; i<$workers; i++)); do
    python trainer_DILAB_local.py \
       worker \
       $i  \
       --worker_num $workers \
       --ps_num $ps \
       &
done


