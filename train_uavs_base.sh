#!/bin/bash

python run_experiment.py --name uav_sim_base --run PPO \
    --env_name "multi-uav-v0"\
    train \
    --cpu 12 \
    --gpu 1 \
    --stop_timesteps 30000000

python run_experiment.py train --run PPO --stop_iters 500 --gpu 1 --num_envs_per_worker 8 --log_dir ./logs

