import argparse
from datetime import datetime
from time import time
from matplotlib import pyplot as plt
import numpy as np
import ray
from ray import air, tune
from uav_sim.envs.uav_sim import UavSim  # Import custom environment
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import get_trainable_cls, register_env
from uav_sim.utils.callbacks import TrainCallback
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
import logging
import json
#from uav_sim.utils.safety_layer import SafetyLayer
#from plot_results import plot_uav_states
import math
import os
from uav_sim.networks.fix_model import TorchFixModel
from uav_sim.networks.cnn_model import TorchCnnModel
from uav_sim.utils.utils import get_git_hash

PATH = Path(__file__).parent.absolute().resolve()
RESULTS_DIR = Path.home() / "ray_results"
logger = logging.getLogger(__name__)

ModelCatalog.register_custom_model("torch_fix_model", TorchFixModel)
ModelCatalog.register_custom_model("torch_cnn_model", TorchCnnModel)

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
    # Use your custom environment to get obs and action space
    env_config = config["env_config"]
    #render = env_config["render"]
    #env_config["render"] = False

    # Create an instance of your custom environment
    temp_env = UavSim(env_config)

    env_obs_space = temp_env.observation_space
    env_action_space = temp_env.action_space

    temp_env.close()
    #env_config["render"] = render

    return env_obs_space, env_action_space


def get_algo_config(config, env_obs_space, env_action_space, env_task_fn=None):
    custom_model = config["exp_config"].setdefault("custom_model", "torch_cnn_model")

    algo_config = (
        get_trainable_cls(config["exp_config"]["run"])
        .get_default_config()
        .environment(
            env=config["env_name"],
            env_config=config["env_config"],
            env_task_fn=env_task_fn,
        )
        .framework(config["exp_config"]["framework"])
        .rollouts(num_rollout_workers=0)
        .debugging(log_level="ERROR", seed=config["env_config"]["seed"])
        .resources(num_gpus=0, num_gpus_per_learner_worker=0)
        .multi_agent(
            policies={
                "shared_policy": (None, env_obs_space, env_action_space, {}),
            },
            # Always use "shared" policy.
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        )
    )

    return algo_config


def train(args):
    num_gpus = int(math.ceil(os.environ.get("RLLIB_NUM_GPUS", args.gpu)))
    ray.init(local_mode=args.local_mode, num_gpus=num_gpus)

    env_obs_space, env_action_space = get_obs_act_space(args.config)

    args.config["env_config"]["num_uavs"] = 6
    args.config["env_config"]["uav_type"] = tune.grid_search(["UAV"])  # Match your UAV type
    args.config["env_config"]["use_safe_action"] = tune.grid_search([False])
    args.config["env_config"]["obstacle_collision_weight"] = 1.0
    args.config["env_config"]["uav_collision_weight"] = 1.0
    args.config["env_config"]["collision_penalty"] = -100
    args.config["env_config"]["stp_penalty"] = tune.grid_search([0.8])

    obs_filter = "NoFilter"
    callback_list = [TrainCallback]

    train_config = (
        get_algo_config(
            args.config, env_obs_space, env_action_space)
        .rollouts(
            num_rollout_workers=1 if args.smoke_test else args.cpu,
            num_envs_per_worker=args.num_envs_per_worker,
            batch_mode="complete_episodes",
            observation_filter=obs_filter,
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
            train_batch_size=4096,
            gamma=0.99,
            num_sgd_iter=32,
            sgd_minibatch_size=64,
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
            storage_path=args.log_dir,
            name=args.name,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=100,
                checkpoint_at_end=True,
                checkpoint_frequency=5,
            ),
        ),
    )

    results = tuner.fit()

def experiment(exp_config={}, max_num_episodes=1, experiment_num=0):
    fname = exp_config.setdefault("fname", None)
    write_experiment = exp_config.setdefault("write_experiment", False)
    env_config = exp_config["env_config"]
    #render = exp_config["render"]
    #plot_results = exp_config["plot_results"]

    algo_to_run = exp_config["exp_config"].setdefault("run", "PPO")
    if algo_to_run not in ["PPO"]:
        print("Unrecognized algorithm. Exiting...")
        exit(99)

    env = UavSim(env_config)

    if algo_to_run == "PPO":
        checkpoint = exp_config["exp_config"].setdefault("checkpoint", None)
        env_obs_space, env_action_space = get_obs_act_space(exp_config)

        if checkpoint is not None:
            use_policy = True
            from ray.rllib.policy.policy import Policy
            from ray.rllib.models.preprocessors import get_preprocessor

            algo = Policy.from_checkpoint(checkpoint)
            prep = get_preprocessor(env_obs_space)(env_obs_space)
        else:
            use_policy = False
            env.close()
            algo = (
                get_algo_config(exp_config, env_obs_space, env_action_space)
            ).build()
            env = algo.workers.local_worker().env

    time_step_list = []
    uav_collision_list = [[] for idx in range(env.num_uavs)]
    obstacle_collision_list = [[] for idx in range(env.num_uavs)]
    uav_done_list = [[] for idx in range(env.num_uavs)]
    uav_done_dt_list = [[] for idx in range(env.num_uavs)]
    uav_done_time_list = [[] for idx in range(env.num_uavs)]
    uav_t_go_list = [[] for idx in range(env.num_uavs)]
    rel_pad_dist = [[] for idx in range(env.num_uavs)]
    rel_pad_vel = [[] for idx in range(env.num_uavs)]
    uav_state = [[] for idx in range(env.num_uavs)]
    uav_reward = [[] for idx in range(env.num_uavs)]
    rel_pad_state = [[] for idx in range(env.num_uavs)]
    obstacle_state = [[] for idx in range(env.max_num_obstacles)]
    target_state = []

    results = {
        "num_episodes": 0.0,
        "uav_collision": 0.0,
        "obs_collision": 0.0,
        "uav_reward": 0.0,
        "uav_done": [[] for idx in range(env.num_uavs)],
        "time_step": [[] for idx in range(env.num_uavs)],
        "uav_sa_sat": [[] for idx in range(env.num_uavs)],
        "episode_time": [],
        "episode_data": {
            "time_step_list": [],
            "uav_collision_list": [],
            "obstacle_collision_list": [],
            "uav_done_list": [],
            "uav_done_dt_list": [],
            "uav_dt_go_list": [],
            "uav_t_go_list": [],
            "rel_pad_dist": [],
            "rel_pad_vel": [],
            "uav_state": [],
            "uav_reward": [],
            "rel_pad_state": [],
            "obstacle_state": [],
            "destination": [],
        },
    }

    num_episodes = 0
    env_out, done = env.reset(), {i.id: False for i in env.uavs.values()}
    obs, info = env_out
    done["__all__"] = False

    logger.debug("running experiment")
    num_episodes = 0
    start_time = time()

    while num_episodes < max_num_episodes:
        actions = {}
        for idx in range(env.num_uavs):
            if algo_to_run == "PPO":
                if use_policy:
                    actions[idx] = algo.compute_single_action(prep.transform(obs[idx]), explore=False)[0]
                else:
                    actions[idx] = algo.compute_single_action(obs[idx], policy_id="shared_policy")

        obs, rew, done, truncated, info = env.step(actions)
        for k, v in info.items():
            results["uav_collision"] += v["uav_collision"]
            results["obs_collision"] += v["obstacle_collision"]
            results["uav_crashed"] += v["uav_crashed"]

        for k, v in rew.items():
            results["uav_reward"] += v

        if num_episodes == 0:
            for k, v in info.items():
                uav_collision_list[k].append(v["uav_collision"])
                obstacle_collision_list[k].append(v["obstacle_collision"])
                uav_done_list[k].append(v["uav_reached"])
                uav_t_go_list[k].append(v["uav_t_go"])
                rel_pad_dist[k].append(v["uav_reached_dest"])
                rel_pad_vel[k].append(v["uav_rel_vel"])
                uav_reward[k].append(rew[k])

            for uav_idx in range(env.num_uavs):
                uav_state[uav_idx].append(env.uavs[uav_idx].state.tolist())
                rel_pad_state[uav_idx].append(env.uavs[uav_idx].pad.state.tolist())

            target_state.append(env.target.state.tolist())
            time_step_list.append(env.time_elapsed)

            for obs_idx in range(env.max_num_obstacles):
                obstacle_state[obs_idx].append(env.obstacles[obs_idx].state.tolist())

        #if render:
            #render()

        if done["__all__"]:
            num_episodes += 1
            for k, v in info.items():
                results["uav_done"][k].append(v["uav_reached_dest"])
            results["num_episodes"] = num_episodes
            results["episode_time"].append(env.time_elapsed)

            if num_episodes <= 1:
                results["episode_data"]["time_step_list"].append(time_step_list)
                results["episode_data"]["uav_collision_list"].append(uav_collision_list)
                results["episode_data"]["obstacle_collision_list"].append(obstacle_collision_list)
                results["episode_data"]["uav_done_list"].append(uav_done_list)
                results["episode_data"]["uav_done_dt_list"].append(uav_done_dt_list)
                results["episode_data"]["uav_dt_go_list"].append(uav_dt_go_list)
                results["episode_data"]["uav_t_go_list"].append(uav_t_go_list)
                results["episode_data"]["rel_pad_dist"].append(rel_pad_dist)
                results["episode_data"]["rel_pad_vel"].append(rel_pad_vel)
                results["episode_data"]["uav_state"].append(uav_state)
                results["episode_data"]["target_state"].append(target_state)
                results["episode_data"]["uav_reward"].append(uav_reward)
                results["episode_data"]["rel_pad_state"].append(rel_pad_state)
                results["episode_data"]["obstacle_state"].append(obstacle_state)

            #if render:
                #im = env.render(mode="rgb_array", done=True)
            #if plot_results:
                #plot_uav_states(results, env_config, num_episodes - 1)

            if num_episodes == max_num_episodes:
                end_time = time() - start_time
                break
            env_out, done = env.reset(), {agent.id: False for agent in env.uavs.values()}
            obs, info = env_out
            done["__all__"] = False

            time_step_list = [[] for idx in range(env.num_uavs)]
            uav_collision_list = [[] for idx in range(env.num_uavs)]
            obstacle_collision_list = [[] for idx in range(env.num_uavs)]
            uav_done_list = [[] for idx in range(env.num_uavs)]
            uav_done_dt_list = [[] for idx in range(env.num_uavs)]
            uav_done_time_list = [[] for idx in range(env.num_uavs)]
            uav_dt_go_list = [[] for idx in range(env.num_uavs)]
            uav_t_go_list = [[] for idx in range(env.num_uavs)]
            rel_pad_dist = [[] for idx in range(env.num_uavs)]
            rel_pad_vel = [[] for idx in range(env.num_uavs)]
            uav_state = [[] for idx in range(env.num_uavs)]
            uav_reward = [[] for idx in range(env.num_uavs)]
            rel_pad_state = [[] for idx in range(env.num_uavs)]
            obstacle_state = [[] for idx in range(env.num_obstacles)]
            target_state = []

    env.close()

    logger.debug("done")

def get_default_env_config(path, config):
    env = UavSim(config["env_config"])

    config["env_config"].update(env.env_config)

    with open(path, "w") as f:
        json.dump(config, f)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", default=f"{PATH}/configs/sim_config.json")
    parser.add_argument("--get_config")
    parser.add_argument("--log_dir", )
    parser.add_argument("--run", type=str, help="The RLlib-registered algorithm to use.")
    parser.add_argument("--tf", type=float)
    parser.add_argument("--name", help="Name of experiment.", default="debug")
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument("--env_name", type=str, default="multi-uav-ren-v0")

    subparsers = parser.add_subparsers(dest="command")
    test_sub = subparsers.add_parser("test")
    test_sub.add_argument("--checkpoint")
    test_sub.add_argument("--uav_type", type=str)
    test_sub.add_argument("--max_num_episodes", type=int, default=1)
    test_sub.add_argument("--experiment_num", type=int, default=0)
    test_sub.add_argument("--render", action="store_true", default=False)
    test_sub.add_argument("--write_exp", action="store_true", default=False)
    test_sub.add_argument("--plot_results", action="store_true", default=False)
    test_sub.add_argument("--tune_run", action="store_true", default=False)
    test_sub.add_argument("--seed")

    #test_sub.set_defaults(func=test)

    train_sub = subparsers.add_parser("train")
    train_sub.add_argument("--smoke_test", action="store_true", help="run quicktest")
    train_sub.add_argument(
        "--stop_iters", type=int, default=5000, help="Number of iterations to train."
    )
    train_sub.add_argument(
        "--stop_timesteps",
        type=int,
        default=int(30e6),
        help="Number of timesteps to train.",
    )
    train_sub.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )
    train_sub.add_argument("--checkpoint", type=str)
    train_sub.add_argument("--gpu", type=float, default=0.0)
    train_sub.add_argument("--num_envs_per_worker", type=int, default=1)
    train_sub.add_argument(
        "--cpu", type=int, default=1, help="num_rollout_workers default is 1"
    )
    train_sub.set_defaults(func=train)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    setup_stream()
    with open(args.load_config, "rt") as f:
        args.config = json.load(f)

    if args.get_config:
        get_default_env_config(args.get_config, args.config)
        return 0

    args.config["env_name"] = args.env_name
    logger.debug(f"config: {args.config}")

    if args.run is not None:
        args.config["exp_config"]["run"] = args.run

    if args.tf is not None:
        args.config["env_config"]["time_final"] = args.tf

    if not args.log_dir:
        branch_hash = get_git_hash()
        num_uavs = args.config["env_config"]["num_uavs"]
        num_obs = args.config["env_config"]["max_num_obstacles"]
        dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_dir = f"{args.func.__name__}/{args.config['exp_config']['run']}/{args.env_name}_{dir_timestamp}_{branch_hash}_{num_uavs}u_{num_obs}o/{args.name}"
        args.log_dir = RESULTS_DIR / log_dir

    args.log_dir = Path(args.log_dir).resolve()
    args.func(args)


if __name__ == "__main__":
    main()