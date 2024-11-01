from typing import Dict
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

class TrainCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
        # Initialize data structures to track metrics for the episode
        episode.user_data["collisions_with_uavs"] = 0
        episode.user_data["collisions_with_obstacles"] = 0
        episode.user_data["collisions_with_cylinders"] = 0
        episode.user_data["out_of_bounds"] = 0
        episode.user_data["reached_destination"] = {}
        episode.user_data["time_to_reach_destination"] = {}
        episode.user_data["rewards"] = {}

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
        # Aggregate data at each step
        info = episode.last_info_for()
        if info:
            # Update collision counters
            episode.user_data["collisions_with_uavs"] += info.get("uav_collision", 0)
            episode.user_data["collisions_with_obstacles"] += info.get("obstacle_collision", 0)
            episode.user_data["collisions_with_cylinders"] += info.get("cylinder_collision", 0)
            # Track if a UAV flies out of the world bounds
            if info.get("out_of_bounds", False):
                episode.user_data["out_of_bounds"] += 1

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
        # Process metrics at the end of each episode
        for agent_id in episode.agent_rewards:
            last_info = episode.last_info_for(agent_id)
            if last_info:
                # Track whether each UAV reached its destination
                episode.user_data["reached_destination"][agent_id] = last_info.get("reached_destination", False)
                # Record the time taken to reach the destination if applicable
                if last_info.get("reached_destination", False):
                    episode.user_data["time_to_reach_destination"][agent_id] = episode.length
                # Aggregate rewards for each UAV
                episode.user_data["rewards"][agent_id] = episode.total_reward

        # Compute and store custom metrics
        episode.custom_metrics["total_collisions_with_uavs"] = episode.user_data["collisions_with_uavs"]
        episode.custom_metrics["total_collisions_with_obstacles"] = episode.user_data["collisions_with_obstacles"]
        episode.custom_metrics["total_collisions_with_cylinders"] = episode.user_data["collisions_with_cylinders"]
        episode.custom_metrics["total_out_of_bounds"] = episode.user_data["out_of_bounds"]
        episode.custom_metrics["rewards_at_episode_end"] = sum(episode.user_data["rewards"].values())

        # Compute averages for metrics where applicable
        episode.custom_metrics["average_time_to_reach_destination"] = np.mean(list(episode.user_data["time_to_reach_destination"].values()))
        episode.custom_metrics["percentage_reached_destination"] = np.mean([1 if v else 0 for v in episode.user_data["reached_destination"].values()])