#!/usr/bin/env bash

#rm nohup.out
workers=2
ps=1
for ((i=0; i<$ps; i++)); do
   python trainer_DILAB_vm.py \
      ps \
      $i \
      --worker_num $workers \
      --ps_num $ps  \
      --ps_hosts=10.155.209.27:2225 \
      --worker_hosts=10.155.208.213:2225,10.155.208.240:2225 \
      &
done


for ((i=0; i<$workers; i++)); do
    python trainer_DILAB_vm.py \
       worker \
       $i  \
       --worker_num $workers \
       --ps_num $ps  \
      --ps_hosts=10.155.209.27:2225 \
      --worker_hosts=10.155.208.213:2225,10.155.208.240:2225 \
       &
done