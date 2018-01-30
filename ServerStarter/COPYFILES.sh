#!/usr/bin/env bash
vms="110.155.208.20, 10.155.208.112, 10.155.208.4"
vmlist=$(echo $vms | tr "," "\n")
for vmip in $vmlist
do
    scp -r ~/A3C/MyDistTest/Distributed_VM_example/ ubuntu@$vmip:~
done