import gym
import gym_rsu

from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

# Create environment
env = gym.make("rsu-v0")

# Instantiate the agent
model = PPO2('MlpPolicy', env)

# Train the agent
print("Beginning model training")
model.learn(total_timesteps=int(2e5))
print("** Done training the model **")

# Save the agent
model.save("ppo_rsu")

# deleting it just to make sure we can load successfully again
del model

# Load the trained agent
model = PPO2.load("ppo_rsu")

# Evaluate the agent
mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    env.render()
