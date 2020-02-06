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
                 lam=0.94,
                 gamma=0.99,
                 tensorboard_log='rsu_agents/ppo_2e6_rsu_tensorboard/')
    # Use this name to save the model
    # parameters after training is done
    save_name = 'rsu_agents/ppo_rsu_2e6'
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
model.learn(total_timesteps=int(2e6))
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
mean_reward, n_steps = evaluate_policy(model, env, n_eval_episodes=10)
print(f'Mean Reward = {round(mean_reward, 4)}')

# Enjoy the trained agent
obs = env.reset()
for _ in range(3):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
