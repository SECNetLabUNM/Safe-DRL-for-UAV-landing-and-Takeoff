import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.callbacks import MemoryTrackingCallbacks
'''
config = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="CartPole-v1")
)

config = PPOConfig().training(lr=tune.grid_search([0.01, 0.001, 0.0001]))

'''

'''
for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")
        
'''

'''
tuner = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(
        stop={"episode_reward_mean": 150},
    ),
    param_space=config,
)

tuner.fit()
'''

'''
# ``Tuner.fit()`` allows setting a custom log directory (other than ``~/ray-results``)
tuner = ray.tune.Tuner(
    "PPO",
    param_space=config,
    run_config=train.RunConfig(
        stop={"episode_reward_mean": 150},
        checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
    ),
)

results = tuner.fit()

# Get the best result based on a particular metric.
best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

# Get the best checkpoint corresponding to the best result.
best_checkpoint = best_result.checkpoint
'''
# Construct a generic config object, specifying values within different
# sub-categories, e.g. "training".
config = (PPOConfig().training(gamma=0.9, lr=0.01)
        .environment(env="CartPole-v1")
        .resources(num_gpus=0)
        .rollouts(num_rollout_workers=0)
        .callbacks(MemoryTrackingCallbacks)
    )
# A config object can be used to construct the respective Algorithm.
rllib_algo = config.build()

# In combination with a tune.grid_search:
config = PPOConfig()
config.training(lr=tune.grid_search([0.01, 0.001]))
# Use `to_dict()` method to get the legacy plain python config dict
# for usage with `tune.Tuner().fit()`.
tuner = tune.Tuner("PPO",
                   run_config=train.RunConfig(
                       stop={"episode_reward_mean": 150},
                   ),
                   param_space=config.to_dict())


tuner.fit()