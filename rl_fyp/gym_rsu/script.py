import sys

import gym
import gym_rsu

from stable_baselines import PPO2, TD3
from stable_baselines.common.evaluation import evaluate_policy

if len(sys.argv) == 1:
    print("Please specify one of the following agents to train with: PPO - DDPG - HIRO")
    exit(0)
elif 'train' == sys.argv[1]:
    # Create environment
    env = gym.make("rsu-v0")

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

    # Train the agent
    print("Beginning model training")
    model.learn(total_timesteps=int(2e6))
    print("** Done training the model **")

    # Save the agent
    model.save(save_name)

    # deleting it just to make sure we can load successfully again
    del model

    # Re-load the trained PPO algorithm with
    # parameters saved as 'ppo_rsu'
    model = PPO2.load(save_name)

    # Evaluate the agent
    mean_reward, n_steps = evaluate_policy(model, env, n_eval_episodes=10)
    print(f'Mean Reward = {round(mean_reward, 4)}')

    # Enjoy the trained agent
    # ------------------------------------------------
    # This has nothing to do with testing the agent in
    # a live simulation. This is just a visualization
    # in the terminal window.
    # ------------------------------------------------
    obs = env.reset()
    for _ in range(3):
        action, _states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        env.render()
elif 'test' == sys.argv[1]:
    # Run the trained agent in the ns3-gym environment
    # to see it in action during simulation
    pass
