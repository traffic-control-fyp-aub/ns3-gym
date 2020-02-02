import sys

import gym
import gym_rsu

from stable_baselines import PPO2, TD3
from stable_baselines.common.evaluation import evaluate_policy

# Create environment
env = gym.make("rsu-v0")

# Doing the below step to get rid of instantiation
# bug that complains about no definition of model variable
model = None
save_name = ''
if len(sys.argv) == 1:
    print("Please specify one of the following agents to train with: PPO - DDPG - HIRO")
    exit(0)
elif 'ppo' == sys.argv[1]:
    # Use the stable-baseline PPO policy to train
    model = PPO2('MlpPolicy',
                 env,
                 verbose=1,
                 ent_coef=0.0,
                 n_steps=2048,
                 nminibatches=32,
                 lam=0.95,
                 gamma=0.99,
                 noptepochs=10,
                 learning_rate=0.0003,
                 cliprange=0.2)
    # Use this name to save the model
    # parameters after training is done
    save_name = 'ppo_rsu'
elif 'td3' == sys.argv[1]:
    # Use the stable-baseline TD3 policy to train
    model = TD3('MlpPolicy', env, verbose=1, random_exploration=0)
    # Use this name to save the model
    # parameters after training is done
    save_name = 'td3_rsu'
elif 'hiro' == sys.argv[1]:
    # Use the custom HIRO policy to train
    pass

# Train the agent
print("Beginning model training")
model.learn(total_timesteps=int(2e5))
print("** Done training the model **")

# Save the agent
model.save(save_name)

# deleting it just to make sure we can load successfully again
del model

model = None
if 'ppo' == sys.argv[1]:
    # Re-load the trained PPO algorithm with
    # parameters saved as 'ppo_rsu'
    model = PPO2.load(save_name)
elif 'td3' == sys.argv[1]:
    # Re-load the trained TD3 algorithm with
    # parameters saved as 'td3_rsu'
    model = TD3.load(save_name)
elif 'hiro' == sys.argv[1]:
    # Re-load the trained HIRO algorithm with
    # parameters saved as 'hiro_rsu'
    pass

# Evaluate the agent
mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    env.render()
