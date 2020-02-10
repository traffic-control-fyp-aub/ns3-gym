import sys

import gym

# ---------------------------------------------------
# RSU environment that we used to train our PPO agent
# in an offline manner
# ---------------------------------------------------
import gym_rsu

# ---------------------------------------------------------------
# ns3 environment that is used as a connection between openAI Gym
# and ns3 to be able to benchmark the performance of the trained
# agent on a live simulation
# ---------------------------------------------------------------
from ns3gym import ns3env

from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

# Use this name to save the model
# parameters after training is done
save_name = 'rsu_agents/ppo_rsu_2e6'

if len(sys.argv) == 1:
    print("Please specify one of the following: [ test | train ]")
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
    # Creating the ns3 environment that will act as a link
    # between our agent and the live simulation
    env = ns3env.Ns3Env(port=5555,
                        stepTime=0.5,
                        startSim=0,
                        simSeed=12,
                        simArgs={"--duration": 10},
                        debug=False)

    env.reset()

    # Collecting the observation and action spaces of the environment
    # Note that these are different from the ones present in the RSU environment
    # These depend on the C++ implementation one follows when creating the ns3
    # side of the simulation environment
    ob_space = env.observation_space
    ac_space = env.action_space

    print("Observation space: ", ob_space, ob_space.dtype)
    print("Action space: ", ac_space, ac_space.dtype)

    stepIdx, currIt = 0, 0

    try:
        while True:
            print("Start iteration: ", currIt)
            obs = env.reset()
            reward = 0
            done = False
            info = None
            print("Step: ", stepIdx)
            print("-- obs: ", obs)

            model = PPO2.load(save_name)

            while True:
                stepIdx += 1
                action, _states = model.predict(obs)

                print("Step: ", stepIdx)
                obs, reward, done, _ = env.step(action)

                print(f'{obs}, {reward}, {done}')

    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")

    finally:
        env.close()
        print("Done")
