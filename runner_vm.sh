#!/usr/bin/env bash
pss="10.155.209.68"
workers="10.155.209.35"
port="2222"

pslist=$(echo $pss | tr "," "\n")
workerlist=$(echo $workers | tr "," "\n")

#prepare Strings for python script args
for vmip in $pslist
do
   psstring=$psstring$vmip":"$port","
done
psstring=${psstring::-1}

for vmip in $workerlist
do
   workerstring=$workerstring$vmip":"$port","
done
workerstring=${workerstring::-1}

#run python scripts remotely
i=0
for vmip in $pslist
do
   ssh ubuntu@$vmip << EOF
   python ~/Distributed_VM_example/trainer_DILAB_vm.py \
      ps \
      $i \
      --ps_hosts=$psstring \
      --worker_hosts=$workerstring \
      &
EOF
   ((i++))
   echo $i
done


i=0
for vmip in $workerlist
do
   ssh ubuntu@$vmip << EOF
   time
   python ~/Distributed_VM_example/trainer_DILAB_vm.py \
      ps \
      $i \
      --ps_hosts=$psstring \
      --worker_hosts=$workerstring \
      &
EOF
   ((i++))
   echo $i
done

