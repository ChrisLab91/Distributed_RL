#!/usr/bin/env bash
python3 trainer.py \
     --ps_hosts=10.155.209.25:2222 \
     --worker_hosts=10.155.208.20:2222,10.155.208.112:2222 \
     --job_name=worker --task_index=0 | ssh ubuntu@10.155.208.20 python3 -