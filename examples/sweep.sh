#!/usr/bin/env bash
# The headline experiment: a mixed workload on two replicas, sweeping concurrent agents for three scheduling policies.
set -euo pipefail
agentsim sweep \
  --profile chat:0.5,react:0.3,fanout:0.2 --runs 300 \
  --gpus 2 --max-batch 32 \
  --agents 4,8,16,32,64,128 --policy fifo,srpt,progress \
  --out results/sweep_mixed.csv --plot results/sweep_mixed.png

# Fan-out cost: how much of a run's time is spent waiting at joins, and whether more replicas help.
for g in 1 2 4 8; do
  agentsim run --profile fanout --runs 200 --gpus "$g" --agents 32 --json "results/fanout_${g}gpu.json" | grep -E "runs_per_min|latency_p95|join_wait_share|usd_per_1k"
done
