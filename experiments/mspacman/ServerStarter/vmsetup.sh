#!/usr/bin/env bash
vms="10.155.209.68, 10.155.209.35" #, 10.155.208.112, 10.155.208.4"
#path="Distributed_VM_example/ServerStarter/vmscript.sh"

vmlist=$(echo $vms | tr "," "\n")
for vmip in $vmlist
do
    scp -r ~/A3C/MyDistTest/Distributed_VM_example/ ubuntu@$vmip:~
done
for vmip in $vmlist
do
    ssh ubuntu@$vmip 'bash Distributed_VM_example/ServerStarter/vmscript.sh'
done

