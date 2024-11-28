from typing import Dict
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomTrainingCallback(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        """
        Called at the start of each episode.
        """
        assert episode.length <= 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        logger.info(f"Episode {episode.episode_id} (env-idx={env_index}) started.")

        # Initialize custom metrics for tracking collisions, goal reach times, etc.
        episode.user_data["obstacle_collisions"] = []
        episode.user_data["uav_collisions"] = []
        episode.user_data["uav_reached_dest"] = []
        episode.user_data["uav_out_of_bounds"] = []
        episode.user_data["time_step"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        """
        Called at each step of the episode.
        """
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        agent_ids = episode.get_agents()

        for agent_id in agent_ids:
            last_info = episode.last_info_for(agent_id)
            if last_info is not None:
                episode.user_data["obstacle_collisions"].append(last_info["obstacle_collision"])
                episode.user_data["uav_collisions"].append(last_info["uav_collision"])
                episode.user_data["uav_reached_dest"].append(last_info["uav_reached_dest"])
                episode.user_data["uav_out_of_bounds"].append(last_info["uav_out_of_bounds"])
                episode.user_data["time_step"].append(last_info["time_step"])

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        """
        Called at the end of each episode.
        """
        agent_ids = episode.get_agents()
        num_agents = len(agent_ids)

        # Compute average metrics for the episode
        obstacle_collisions = np.sum(episode.user_data["obstacle_collisions"]) / num_agents
        uav_collisions = np.sum(episode.user_data["uav_collisions"]) / num_agents
        uav_reached_dest = np.sum(episode.user_data["uav_reached_dest"]) / num_agents
        uav_out_of_bounds = np.sum(episode.user_data["uav_out_of_bounds"]) / num_agents
        avg_time_step = np.mean(episode.user_data["time_step"]) if episode.user_data["time_step"] else 0.0

        # Log custom metrics
        episode.custom_metrics["obstacle_collisions"] = obstacle_collisions
        episode.custom_metrics["uav_collisions"] = uav_collisions
        episode.custom_metrics["uav_reached_dest"] = uav_reached_dest
        episode.custom_metrics["uav_out_of_bounds"] = uav_out_of_bounds
        episode.custom_metrics["avg_time_step"] = avg_time_step

        logger.info(
            f"Episode {episode.episode_id} ended: Obstacle Collisions = {obstacle_collisions}, "
            f"UAV Collisions = {uav_collisions}, UAV Reached Destination = {uav_reached_dest}, "
            f"UAV Out of Bounds = {uav_out_of_bounds}, Average Time Step = {avg_time_step}"
        )
