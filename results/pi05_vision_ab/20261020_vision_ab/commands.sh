#!/usr/bin/env bash
set -euo pipefail
cd /root/nfs/embodied/benchmark
python /root/nfs/embodied/benchmark/experiments/pi05_vision_ab/run.py --stage audit --run-id 20261020_vision_ab
python /root/nfs/embodied/benchmark/experiments/pi05_vision_ab/run.py --stage check --run-id 20261020_vision_ab --gpu 0
python /root/nfs/embodied/benchmark/experiments/pi05_vision_ab/run.py --stage collect_obs --run-id 20261020_vision_ab --gpu 0
python /root/nfs/embodied/benchmark/experiments/pi05_vision_ab/run.py --stage bench --run-id 20261020_vision_ab --gpu 0
python /root/nfs/embodied/benchmark/experiments/pi05_vision_ab/run.py --stage smoke --run-id 20261020_vision_ab --gpus 0
python /root/nfs/embodied/benchmark/experiments/pi05_vision_ab/run.py --stage screen --run-id 20261020_vision_ab --gpus 0,2,3
python /root/nfs/embodied/benchmark/experiments/pi05_vision_ab/run.py --stage select --run-id 20261020_vision_ab
python /root/nfs/embodied/benchmark/experiments/pi05_vision_ab/run.py --stage confirm --run-id 20261020_vision_ab --gpus 0,2,3
python /root/nfs/embodied/benchmark/experiments/pi05_vision_ab/run.py --stage report --run-id 20261020_vision_ab
