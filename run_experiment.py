import argparse
from datetime import datetime
from time import time
import numpy as np
import ray
from ray import air, tune
from pathlib import Path
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import get_trainable_cls
from uav_sim.utils.callbacks import CustomTrainingCallback
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.rllib.models import ModelCatalog
from uav_sim.envs.uav_sim import UavSim
import os
import logging
import json
import math

PATH = Path(__file__).parent.absolute().resolve()
RESULTS_DIR = Path.home() / "ray_results"
logger = logging.getLogger(__name__)

def env_creator(env_config):
    return UavSim(env_config)

register_env("multi-uav-v0", env_creator)

def setup_stream(logging_level=logging.DEBUG):
    # Turns on logging to console
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "<%(module)s:%(funcName)s:%(lineno)s> - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging_level)


def get_obs_act_space(config):
    # Create a temporary environment to get obs and action space
    env_config = config["env_config"]
    env_config["render"] = False
    temp_env = UavSim(env_config)
    env_obs_space = temp_env.observation_space
    env_action_space = temp_env.action_space
    temp_env.close()
    return env_obs_space, env_action_space


def get_algo_config(config, env_obs_space, env_action_space):
    algo_config = (
        get_trainable_cls(config["exp_config"]["run"])
        .get_default_config()
        .environment(
            env=config["env_name"],
            env_config=config["env_config"]
        )
        .framework(config["exp_config"]["framework"])
        .rollouts(num_rollout_workers=0)
        .debugging(log_level="ERROR", seed=config["env_config"]["seed"])
        .resources(num_gpus=0)
        .multi_agent(
            policies={
                "shared_policy": (
                    None,
                    env_obs_space,
                    env_action_space,
                    {},
                )
            },
            policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "shared_policy"),
        )
    )
    return algo_config


def train(args):
    num_gpus = int(math.ceil(os.environ.get("RLLIB_NUM_GPUS", args.gpu)))
    ray.init(local_mode=args.local_mode, num_gpus=num_gpus)

    # Get the spaces here before varying the experiment treatments (factors)
    env_obs_space, env_action_space = get_obs_act_space(args.config)

    # Vary treatments here
    #args.config["env_config"].update({
    #    "num_uavs": 2,
    #    "num_obstacles": 4,
    #    "uav_type": tune.grid_search(["UAV"]),
    #    "max_time": 50.0,
    #    "obstacle_radius": 0.1,
    #    "max_velocity": 0.5,
    #     "max_acceleration": 0.5,
    #     "reward_params": {
    #         "destination_reached_reward": 500.0,
    #         "collision_penalty": -100.0,
    #         "out_of_bounds_penalty": -100.0,
    #         "time_step_penalty": -2.0,
    #         "towards_dest_reward": 1.0,
    #         "exceed_acc_penalty": -1.0,
    #         "exceed_vel_penalty": -1.0
    #     }
    # })

    task_fn = None
    callback_list = [CustomTrainingCallback]

    # Configure the training algorithm
    train_config = (
        get_algo_config(args.config, env_obs_space, env_action_space)
        .rollouts(
            num_rollout_workers=(1 if args.smoke_test else args.cpu),
            num_envs_per_worker=args.num_envs_per_worker,
            batch_mode="complete_episodes",
            observation_filter="NoFilter"
        )
        .resources(
            num_gpus=0 if args.smoke_test else num_gpus,
            num_learner_workers=1,
            num_gpus_per_learner_worker=0 if args.smoke_test else args.gpu,
        )
        .training(
            lr=5e-5,
            use_gae=True,
            use_critic=True,
            lambda_=0.95,
            train_batch_size=8192,
            gamma=0.99,
            num_sgd_iter=32,
            sgd_minibatch_size=4096,
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            clip_param=0.2,
            grad_clip=1.0,
        )
    )

    multi_callbacks = make_multi_callbacks(callback_list)
    train_config.callbacks(multi_callbacks)

    stop = {
        "training_iteration": 1 if args.smoke_test else args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    tuner = tune.Tuner(
        args.config["exp_config"]["run"],
        param_space=train_config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            local_dir=args.log_dir,
            name=args.name,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=100,
                checkpoint_at_end=True,
                checkpoint_frequency=20,
            ),
        ),
    )

    tuner.fit()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", default=f"{PATH}/configs/sim_config.json")
    parser.add_argument("--log_dir")
    parser.add_argument("--run", type=str, help="The RLlib-registered algorithm to use.")
    parser.add_argument("--name", help="Name of experiment.", default="debug")
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument("--env_name", type=str, default="multi-uav-v0")

    subparsers = parser.add_subparsers(dest="command")
    train_sub = subparsers.add_parser("train")
    train_sub.add_argument("--smoke_test", action="store_true", help="run quicktest")
    train_sub.add_argument("--stop_iters", type=int, default=5000, help="Number of iterations to train.")
    train_sub.add_argument("--stop_timesteps", type=int, default=int(30e6), help="Number of timesteps to train.")
    train_sub.add_argument("--local-mode", action="store_true", help="Init Ray in local mode for easier debugging.")
    train_sub.add_argument("--gpu", type=float, default=1.0)
    train_sub.add_argument("--num_envs_per_worker", type=int, default=1)
    train_sub.add_argument("--cpu", type=int, default=1, help="num_rollout_workers default is 1")
    train_sub.set_defaults(func=train)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    setup_stream()

    with open(args.load_config, "rt") as f:
        args.config = json.load(f)

    args.config["env_name"] = args.env_name

    if args.run is not None:
        args.config["exp_config"]["run"] = args.run

    if not args.log_dir:
        num_uavs = args.config["env_config"]["num_uavs"]
        num_obs = args.config["env_config"]["num_obstacles"]
        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_dir = f"{args.func.__name__}/{args.config['exp_config']['run']}/{args.env_name}_{dir_timestamp}_{num_uavs}u_{num_obs}o/{args.name}"
        args.log_dir = RESULTS_DIR / log_dir

    args.log_dir = Path(args.log_dir).resolve()
    args.func(args)


if __name__ == "__main__":
    main()
