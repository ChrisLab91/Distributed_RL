#!/usr/bin/env bash
#vms= "10.155.208.243, 10.155.208.247, 10.155.208.248, 10.155.208.224, 10.155.208.240, 10.155.208.236, 10.155.209.25"
#vms="10.155.208.163, 10.155.208.229, 10.155.208.117, 10.155.208.129"
#vms="10.155.209.25, 10.155.208.189, 10.155.208.242, 10.155.209.29"
#path="Distributed_VM_example/ServerStarter/vmscript.sh"

vmlist=$(echo $vms | tr "," "\n")
for vmip in $vmlist
do
    scp -r ~/A3C/MyDistTest/for_vm/ ubuntu@$vmip:~
done
for vmip in $vmlist
do
    ssh ubuntu@$vmip 'bash for_vm/ServerStarter/vmscript.sh'
done


