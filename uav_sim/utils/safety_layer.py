import os
from datetime import datetime
from pathlib import Path
import random
from time import time

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from torch import nn
from uav_sim.utils.replay_buffer import ReplayBuffer
from torch.optim import Adam
import ray.tune as tune
from ray.air import Checkpoint, session
from ray.rllib.algorithms.algorithm import Algorithm
from uav_sim.envs import UavSim
from ray.rllib.algorithms.ppo import PPOConfig

ENV_VARIABLES = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONWARNINGS": "ignore::DeprecationWarning",
}
my_runtime_env = {"env_vars": ENV_VARIABLES}
# ray.init(runtime_env=my_runtime_env)
# import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning)


# This is need to ensure reproducibility. See: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

PATH = os.path.dirname(os.path.abspath(__file__))
torch.use_deterministic_algorithms(True)


class SafetyLayer:
    def __init__(self, env, config={}):
        self._env = env
        self._config = config

        self._parse_config()
        self._set_seed()

        self._init_model()

        self._cbf_optimizer = Adam(
            self._cbf_model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )

        self._nn_action_optimizer = Adam(
            self._nn_action_model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        if self._checkpoint_dir:
            checkpoint_state = torch.load(
                self._checkpoint_dir, map_location=torch.device(self._device)
            )
            self._cbf_model.load_state_dict(checkpoint_state["cbf_state_dict"])
            self._nn_action_model.load_state_dict(
                checkpoint_state["nn_action_state_dict"]
            )

            self._cbf_optimizer.load_state_dict(
                checkpoint_state["cbf_optimizer_state_dict"]
            )
            self._nn_action_optimizer.load_state_dict(
                checkpoint_state["nn_action_optimizer_state_dict"]
            )

        if self._load_buffer:
            pass

        self._replay_buffer = ReplayBuffer(self._replay_buffer_size)

        if self._checkpoint:
            checkpoint_state = self._checkpoint.to_dict()
            self._train_global_step = checkpoint_state["epoch"]
            self._cbf_model.load_state_dict(checkpoint_state["cbf_state_dict"])
            self._nn_action_model.load_state_dict(
                checkpoint_state["nn_action_state_dict"]
            )

            self._cbf_optimizer.load_state_dict(
                checkpoint_state["cbf_optimizer_state_dict"]
            )
            self._nn_action_optimizer.load_state_dict(
                checkpoint_state["nn_action_optimizer_state_dict"]
            )
        else:
            self._train_global_step = 0

        # use gpu if available
        # https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
        # self._device = "cpu"
        if torch.cuda.is_available() and self._device == "cuda":
            print("using cuda")
            self._device = "cuda"
        self._cbf_model.to(self._device)
        self._nn_action_model.to(self._device)

    def _set_seed(self):
        """Assumes random, np.random.seed has already been seeded from environment."""
        # random.seed(self._seed)
        # np.random.seed(self._seed)
        torch.manual_seed(self._seed)

    def _parse_config(self):
        self._replay_buffer_size = self._config.get("replay_buffer_size", 1000000)
        self._episode_length = self._config.get("episode_length", 400)
        self._lr = self._config.get("lr", 0.01)
        self._weight_decay = self._config.get("weight_decay", 1e-5)
        self._batch_size = self._config.get("batch_size", 64)
        self._num_eval_steps = self._config.get("num_eval_steps", 1500)
        self._num_training_steps = self._config.get("num_training_steps", 6000)
        self._num_epochs = self._config.get("num_epochs", 25)
        self._num_iter_per_epoch = self._config.get(
            "num_iter_per_epoch",
            self._num_training_steps * self._env.num_uavs // self._batch_size,
        )
        self._tune_run = self._config.get("tune_run", False)
        self._seed = self._config.get("seed", 123)
        self._checkpoint_dir = self._config.get("checkpoint_dir", None)
        self._log_dir = self._config.get("log_dir", None)
        self._checkpoint_freq = self._config.get("checkpoint_freq", 5)
        self._checkpoint = self._config.get("checkpoint", None)
        self._load_buffer = self._config.get("buffer", None)
        self._n_hidden = self._config.get("n_hidden", 32)
        self.eps_safe = self._config.get("eps_safe", 0.001)
        self.eps_dang = self._config.get("eps_dang", 0.05)
        self.eps_action = self._config.get("eps_action", 0.0)
        self.eps_deriv_safe = self._config.get("eps_deriv_safe", 0.0)
        self.eps_deriv_dang = self._config.get("eps_deriv_dang", 8e-2)
        self.eps_deriv_mid = self._config.get("eps_deriv_mid", 3e-2)
        self.loss_action_weight = self._config.get("loss_action_weight", 0.08)
        self._device = self._config.get("device", "cpu")
        self._safe_margin = self._config.get("safe_margin", 0.1)
        self._unsafe_margin = self._config.get("unsafe_margin", 0.01)
        self._use_rl = self._config.get("use_rl", False)

    def _init_model(self):
        obs_space = self._env.observation_space
        n_state = obs_space[0]["state"].shape[0]
        n_rel_pad_state = obs_space[0]["rel_pad"].shape[0]
        self.k_obstacle = obs_space[0]["obstacles"].shape[1]  # (num_obstacle, state)
        m_control = self._env.action_space[0].shape[0]

        self._cbf_model = CBF(n_state=self.k_obstacle, n_hidden=self._n_hidden)
        num_o = (
            obs_space[0]["obstacles"].shape[0] + obs_space[0]["other_uav_obs"].shape[0]
        )
        self._nn_action_model = NN_Action(
            n_state=self.k_obstacle,
            m_control=m_control,
            n_hidden=self._n_hidden,
            num_o=num_o,
            # n_state, n_rel_pad_state, k_obstacle, m_control, self._n_hidden
        )

        if self._use_rl:
            tune.register_env(
                "multi-uav-sim-v0",
                # lambda env_config: UavSim(env_config=env_config),
                lambda env_config: self._env,
            )

            self._algo = (
                PPOConfig()
                # .environment(
                #     env=exp_config["env_name"], env_config=exp_config["env_config"]
                # )
                .framework("torch")
                .rollouts(num_rollout_workers=0)
                .debugging(log_level="ERROR", seed=self._seed)
                .resources(num_gpus=0)
                .multi_agent(
                    policies={
                        "shared_policy": (
                            None,
                            self._env.observation_space[0],
                            self._env.action_space[0],
                            {},
                        )
                    },
                    # Always use "shared" policy.
                    policy_mapping_fn=(
                        lambda agent_id, episode, worker, **kwargs: "shared_policy"
                    ),
                )
                .build()
            )

            self._algo.restore(
                "/home/prime/Documents/workspace/rl_multi_uav_sim/results/PPO/multi-uav-sim-v0_2023-11-06-23-23_e7633c3/cur_col_01/PPO_multi-uav-sim-v0_6dfd0_00001_1_obstacle_collision_weight=0.1000,stp_penalty=5,uav_collision_weight=0.1000,use_safe_action=Fals_2023-11-06_23-23-40/checkpoint_000301"
            )

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        tensor = tensor.to(self._device)
        return tensor


    def _sample_steps(self, num_steps):
        episode_length = 0
        num_episodes = 0

        results = {
            "uav_collision": 0.0,
            "obs_collision": 0.0,
            "uav_rel_dist": 0.0,
            "uav_done": 0.0,
            "uav_done_dt": 0.0,
        }

        obs, info = self._env.reset()

        for step_num in range(num_steps):
            # print(f"step num: {step_num}")
            nom_actions = {}
            actions = {}
            for i in range(self._env.num_uavs):
                if self._use_rl:
                    nom_actions[i] = self._algo.compute_single_action(
                        obs[i], policy_id="shared_policy"
                    ).squeeze()
                else:
                    nom_actions[i] = self._env.get_time_coord_action(
                        self._env.uavs[i]
                    ).squeeze()

                actions[i] = self.get_action(obs[i], nom_actions[i])

            obs_next, _, done, truncated, info = self._env.step(actions)

            for k, v in info.items():
                results["uav_collision"] += v["uav_collision"]
                results["obs_collision"] += v["obstacle_collision"]

            for i in range(self._env.num_uavs):
                buffer_dictionary = {}
                for k, v in obs[i].items():
                    buffer_dictionary[k] = v

                buffer_dictionary["u_nominal"] = nom_actions[i]

                for k, v in obs_next[i].items():
                    new_key = f"{k}_next"
                    buffer_dictionary[new_key] = v
                self._replay_buffer.add(buffer_dictionary)

            obs = obs_next
            episode_length += 1

            # self._env.render()

            if done["__all__"] or (episode_length == self._episode_length):
                num_episodes += 1
                for k, v in info.items():
                    results["uav_rel_dist"] += v["uav_rel_dist"]
                    results["uav_done"] += v["uav_landed"]
                    results["uav_done_dt"] += v["uav_done_dt"]

                obs, info = self._env.reset()
                episode_length = 0

        num_episodes += 1
        results["uav_rel_dist"] = (
            results["uav_rel_dist"] / num_episodes / self._env.num_uavs
        )
        results["obs_collision"] = (
            results["obs_collision"] / num_episodes / self._env.num_uavs
        )
        results["uav_collision"] = (
            results["uav_collision"] / num_episodes / self._env.num_uavs
        )
        results["uav_done"] = results["uav_done"] / num_episodes / self._env.num_uavs
        results["uav_done_dt"] = (
            results["uav_done_dt"] / num_episodes / self._env.num_uavs
        )
        results["num_ts_per_episode"] = num_steps / num_episodes
        return results

    def _get_mask(self, constraints):
        safe_mask = torch.all((constraints >= self._safe_margin), dim=1).float()
        unsafe_mask = torch.any((constraints <= self._unsafe_margin), dim=1).float()

        # safe_mask = (constraints >= self._safe_margin).float()
        # unsafe_mask = (constraints <= self._unsafe_margin).float()

        mid_mask = (1 - safe_mask) * (1 - unsafe_mask)

        return safe_mask, unsafe_mask, mid_mask

    def f_dot_torch(self, state, action):
        A = np.zeros((12, 12), dtype=np.float32)
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0

        B = np.zeros((12, 3), dtype=np.float32)
        B[3, 0] = 1.0
        B[4, 1] = 1.0
        B[5, 2] = 1.0
        A_T = self._as_tensor(A.T)
        B_T = self._as_tensor(B.T)

        dxdt = torch.matmul(state, A_T) + torch.matmul(action, B_T)

        return dxdt

    def _evaluate_batch(self, batch):
        """Gets the observation and calculate h and action from model.

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        state = self._as_tensor(batch["state"])
        rel_pad = self._as_tensor(batch["rel_pad"])
        other_uav_obs = self._as_tensor(batch["other_uav_obs"])
        obstacles = self._as_tensor(batch["obstacles"])
        constraints = self._as_tensor(batch["constraint"])
        u_nominal = self._as_tensor(batch["u_nominal"])
        state_next = self._as_tensor(batch["state_next"])
        rel_pad_next = self._as_tensor(batch["rel_pad_next"])
        other_uav_obs_next = self._as_tensor(batch["other_uav_obs_next"])
        obstacles_next = self._as_tensor(batch["obstacles_next"])

        safe_mask, unsafe_mask, mid_mask = self._get_mask(constraints)

        h = self._cbf_model(state, other_uav_obs, obstacles)
        u = self._nn_action_model(state, rel_pad, other_uav_obs, obstacles, u_nominal)

        # calculate the the nomimal state using https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9681233
        state_next_nominal = state + self.f_dot_torch(state, u) * self._env.dt

        state_next_grad = (
            state_next_nominal + (state_next - state_next_nominal).detach()
        )

        h_next = self._cbf_model(
            state_next_grad,
            # state_next_nominal,
            other_uav_obs_next,
            obstacles_next,
        )
        h_deriv = (h_next - h) / self._env.dt + h

        num_safe = torch.sum(safe_mask)
        num_unsafe = torch.sum(unsafe_mask)
        num_mid = torch.sum(mid_mask)

        loss_h_safe = torch.sum(F.relu(self.eps_safe - h) * safe_mask) / (
            1e-5 + num_safe
        )
        loss_h_dang = torch.sum(F.relu(h + self.eps_dang) * unsafe_mask) / (
            1e-5 + num_unsafe
        )

        acc_h_safe = torch.sum((h >= 0).float() * safe_mask) / (1e-5 + num_safe)
        acc_h_dang = torch.sum((h < 0).float() * unsafe_mask) / (1e-5 + num_unsafe)

        loss_deriv_safe = torch.sum(
            F.relu(self.eps_deriv_safe - h_deriv) * safe_mask
        ) / (1e-5 + num_safe)
        loss_deriv_dang = torch.sum(
            F.relu(self.eps_deriv_dang - h_deriv) * unsafe_mask
        ) / (1e-5 + num_unsafe)
        loss_deriv_mid = torch.sum(F.relu(self.eps_deriv_mid - h_deriv) * mid_mask) / (
            1e-5 + num_mid
        )

        acc_deriv_safe = torch.sum((h_deriv >= 0).float() * safe_mask) / (
            1e-5 + num_safe
        )
        acc_deriv_dang = torch.sum((h_deriv >= 0).float() * unsafe_mask) / (
            1e-5 + num_unsafe
        )
        acc_deriv_mid = torch.sum((h_deriv >= 0).float() * mid_mask) / (1e-5 + num_mid)

        err_action = torch.mean(torch.abs(u - u_nominal))

        loss_action = torch.mean(F.relu(torch.abs(u - u_nominal) - self.eps_action))
        # loss_action = torch.sum(
        #     F.relu(torch.linalg.vector_norm(u - u_nominal, dim=-1) - self.eps_action) * torch.min(safe_mask, dim=-1)[0]
        # ) / (1e-5 + num_safe)

        # loss = (1 / (1 + self.loss_action_weight)) * (
        loss = (
            loss_h_safe
            + 3.0 * loss_h_dang
            + loss_deriv_safe
            + 3.0 * loss_deriv_dang
            + 2.0 * loss_deriv_mid
            + self.loss_action_weight * loss_action
        )
        # ) + loss_action * self.loss_action_weight / (1 + self.loss_action_weight)


        return loss, (
            acc_h_safe.detach().cpu().numpy(),
            acc_h_dang.detach().cpu().numpy(),
            acc_deriv_safe.detach().cpu().numpy(),
            acc_deriv_dang.detach().cpu().numpy(),
            acc_deriv_mid.detach().cpu().numpy(),
            err_action.detach().cpu().numpy(),
        )

    def _train_batch(self):
        """Sample batch from replay buffer and calculate loss

        Returns:
            loss function
        """
        batch = self._replay_buffer.sample(self._batch_size)

        # forward + backward + optimize
        loss, acc_stats = self._evaluate_batch(batch)

        # zero parameter gradients
        self._nn_action_optimizer.zero_grad()
        self._cbf_optimizer.zero_grad()

        loss.backward()

        self._nn_action_optimizer.step()
        self._cbf_optimizer.step()

        return loss, acc_stats

    def parse_results(self, results):
        loss_array = []
        acc_stat_array = []
        for x in results:
            loss_array.append(x[0].item())
            acc_stat_array.append([acc for acc in x[1]])

        loss = np.array(loss_array).mean()
        acc_stat = np.array(acc_stat_array).mean(axis=0)

        return loss, acc_stat

    def evaluate(self):
        """Validation Step"""
        # sample steps
        sample_stats = self._sample_steps(self._num_eval_steps)

        self._cbf_model.eval()
        self._nn_action_model.eval()

        eval_results = [
            self._evaluate_batch(batch)
            for batch in self._replay_buffer.get_sequential(self._batch_size)
        ]

        loss, acc_stat = self.parse_results(eval_results)

        self._replay_buffer.clear()

        self._cbf_model.train()
        self._nn_action_model.train()

        sample_stats = {f"val_{k}": v for k, v in sample_stats.items()}
        return loss, acc_stat, sample_stats

    def get_action(self, obs, action):
        """
        See faq on need to disable both:
        https://stackoverflow.com/questions/55627780/evaluating-pytorch-models-with-torch-no-grad-vs-model-eval

        Args:
            obs (_type_): _description_
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        state = torch.unsqueeze(self._as_tensor(obs["state"]), dim=0)
        rel_pad = torch.unsqueeze(self._as_tensor(obs["rel_pad"]), dim=0)
        other_uav_obs = torch.unsqueeze(self._as_tensor(obs["other_uav_obs"]), dim=0)
        obstacles = torch.unsqueeze(self._as_tensor(obs["obstacles"]), dim=0)
        constraint = torch.unsqueeze(self._as_tensor(obs["constraint"]), dim=0)
        u_nominal = torch.unsqueeze(self._as_tensor(action.squeeze()), dim=0)

        self._cbf_model.eval()
        self._nn_action_model.eval()

        with torch.no_grad():
            u = self._nn_action_model(
                state, rel_pad, other_uav_obs, obstacles, u_nominal
            )

            return u.detach().cpu().numpy().squeeze()

    def fit(self):
        sample_stats = self._sample_steps(self._num_training_steps)

        # make sure we're in training mode
        self._cbf_model.train()
        self._nn_action_model.train()

        # iterate through the buffer and get batches at a time
        train_results = [self._train_batch() for _ in range(self._num_iter_per_epoch)]

        loss, train_acc_stats = self.parse_results(train_results)
        # self._replay_buffer.clear()
        self._train_global_step += 1

        sample_stats = {f"train_{k}": v for k, v in sample_stats.items()}
        return loss, train_acc_stats, sample_stats

    def train(self):
        """Train Step"""
        start_time = time()

        print("==========================================================")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        for epoch in range(self._num_epochs):
            train_loss, train_acc_stats, train_sample_stats = self.fit()
            print(
                f"Finished training epoch {epoch} with loss: {train_loss}.\n\tstats: {train_sample_stats}.\n\t{train_acc_stats}"
            )

            # print("Running validation:")
            # val_loss, val_acc_stats, val_sample_stats = self.evaluate()
            # print(
            #     f"Validation completed, average loss {val_loss}. val stats: {val_sample_stats}"
            # )

            if self._tune_run:
                train_val_stats = {}
                train_val_stats["train_loss"] = train_loss
                train_val_stats["train_acc_h_safe"] = train_acc_stats[0]
                train_val_stats["train_acc_h_dang"] = train_acc_stats[1]
                train_val_stats["train_acc_h_deriv_safe"] = train_acc_stats[2]
                train_val_stats["train_acc_h_deriv_dang"] = train_acc_stats[3]
                train_val_stats["train_acc_h_deriv_mid"] = train_acc_stats[4]
                train_val_stats["train_err_action"] = train_acc_stats[5]
                # train_val_stats["val_loss"] = val_loss
                # train_val_stats["val_acc_h_safe"] = val_acc_stats[0]
                # train_val_stats["val_acc_h_dang"] = val_acc_stats[1]
                # train_val_stats["val_acc_h_deriv_safe"] = val_acc_stats[2]
                # train_val_stats["val_acc_h_deriv_dang"] = val_acc_stats[3]
                # train_val_stats["val_acc_h_deriv_mid"] = val_acc_stats[4]
                # train_val_stats["val_err_action"] = val_acc_stats[5]

                train_val_stats.update(train_sample_stats)
                # train_val_stats.update(val_sample_stats)
                tune.report(**train_val_stats)

                if (epoch + 1) % self._checkpoint_freq == 0:
                    with tune.checkpoint_dir(epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")

                        torch.save(
                            {
                                "cbf_state_dict": self._cbf_model.state_dict(),
                                "nn_action_state_dict": self._nn_action_model.state_dict(),
                                "cbf_optimizer_state_dict": self._cbf_optimizer.state_dict(),
                                "nn_action_optimizer_state_dict": self._nn_action_optimizer.state_dict(),
                            },
                            path,
                        )
            # else:
            elif self._log_dir is not None:
                checkpoint_dir = (Path(self._log_dir) / f"checkpoint_{epoch}").resolve()
                if not checkpoint_dir.exists():
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)
                if (
                    epoch + 1
                ) % self._checkpoint_freq == 0 and checkpoint_dir is not None:
                    path = checkpoint_dir / "checkpoint"

                    torch.save(
                        {
                            "cbf_state_dict": self._cbf_model.state_dict(),
                            "nn_action_state_dict": self._nn_action_model.state_dict(),
                            "cbf_optimizer_state_dict": self._cbf_optimizer.state_dict(),
                            "nn_action_optimizer_state_dict": self._nn_action_optimizer.state_dict(),
                        },
                        path,
                    )



                #     checkpoint_data = {
                #         "epoch": self._train_global_step,
                #         "net_state_dict": self.model.state_dict(),
                #         "optimizer_state_dict": self._optimizer.state_dict(),
                #     }
                #     checkpoint = Checkpoint.from_dict(checkpoint_data)

                #     session.report(train_val_stats, checkpoint=checkpoint)
                # else:
                #     session.report(
                #         train_val_stats,
                #     )

        print("==========================================================")
        print(
            f"Finished training constraint model. Time spent: {(time() - start_time) // 1} secs"
        )
        print("==========================================================")
