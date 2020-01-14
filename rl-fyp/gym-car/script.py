import gym

import rl-fyp.rl-baselines.stable-baselines.stable_baselines import DQN
from rl-fyp.rl-baselines.stable-baselines.stable_baselines.common.evaluation import evaluate_policy

import gym-car

# Create environment
env = gym.make('car-v0')

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

# Train the agent
model.learn(total_timesteps=int(2e5))

# Save the agent
model.save("dqn_car")

del model # deleting it just to make sure we can load successfully again

# Load the trained agent
model = DQN.load("dqn_car")

# Evaluate the agent
mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    env.render()
