#!/bin/bash
cd /root/nfs/embodied/benchmark
/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python -m preparations.idea2.cewp_rpc_smoke --output /root/nfs/embodied/benchmark/results/idea2/cewp_20261021/server_rpc_gpu0_rerun --gpu 0 --port 19012 --bank results/idea1_v3_1/input_bank.pkl
