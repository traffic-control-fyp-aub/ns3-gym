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
from stable_baselines.common.vec_env import DummyVecEnv

# Collect the list of command line arguments passed
argumentList = sys.argv

ns3_obj = None


def make_ns3_env():
    """
        This function is created as a bug fix to train
        our PPO agent which requires that all its environments
        be wrapped in a vectorized environment layer.
    """
    return ns3_obj


if argumentList.__len__() == 1:
    # User did not specify the file properly
    print("Please specify one of the following: [ test | train ]"
          " and if you specified to train then [ --online | --offline ]")
    exit(0)
elif argumentList.__len__() is 2:
    if sys.argv[1] in ['test']:
        # Load the previously trained agent parameters and start
        # running the traffic simulation
        # Creating the ns3 environment that will act as a link
        # between our agent and the live simulation
        env = ns3env.Ns3Env(port=5555,
                            stepTime=0.5,
                            startSim=0,
                            simSeed=12,
                            simArgs={"--duration": 10},
                            debug=False)

        ob_space = env.observation_space
        ac_space = env.action_space

        print("Observation Space: ", ob_space, ob_space.dtype)
        print("Action Space: ", ac_space, ac_space.dtype)

        stepIdx, currIt = 0, 0

        try:
            model = PPO2.load('rsu_agents/ppo_offline_weights')  # FIXME - change this name accordingly
            while True:
                print("Start iteration: ", currIt)
                obs = env.reset()
                reward = 0
                done = False
                info = None
                print("Step: ", stepIdx)
                print("-- obs: ", obs)

                while True:
                    stepIdx += 1
                    action, _states = model.predict(obs)
                    print("Predicted action: ", action, type(action))

                    print("Step: ", stepIdx)
                    obs, reward, done, _ = env.step(action)

                    print(f'{obs}, {reward}, {done}')

        except KeyboardInterrupt:
            env.close()
            print("Ctrl-C -> Exit")

        finally:
            env.close()
            print("Done")
    elif sys.argv[1] in ['train']:
        # Raise and exception because the user needs to specify
        # whether the training needs to be online or offline.
        # Online means running it directly in the ns3 environment
        # with SUMO and offline means running it with the RSU custom
        # gym environment.
        print("Please specify one of the following training methods: [ online | offline ]")
        exit(0)
elif argumentList.__len__() is 3:
    if sys.argv[1] in ['train'] and sys.argv[2] in ['online']:
        # Train using the ns3 SUMO environment
        # Creating the ns3 environment that will act as a link
        # between our agent and the live simulation
        ns3_obj = ns3env.Ns3Env(port=5555,
                                stepTime=0.5,
                                startSim=0,
                                simSeed=12,
                                debug=True)

        ob_space = ns3_obj.observation_space
        ac_space = ns3_obj.action_space

        # Vectorized environment to be able to set it to PPO
        env = DummyVecEnv([make_ns3_env])

        print("Observation Space: ", ob_space, ob_space.dtype)
        print("Action Space: ", ac_space, ac_space.dtype)

        stepIdx, currIt = 0, 0

        try:
            print('Setting up the PPO model')
            # Use the stable-baseline PPO policy to train
            model = PPO2('MlpPolicy',
                         env=env,
                         learning_rate=3e-4,
                         verbose=1,
                         ent_coef=0.0,
                         lam=0.94,
                         gamma=0.99,
                         tensorboard_log='rsu_agents/ppo_online_tensorboard/')

            print('Training model')
            # Start the learning process on the ns3 + SUMO environment
            model.learn(total_timesteps=int(1e3))
            print(' ** Done Training ** ')

            print('Saving Model')
            model.save('rsu_agents/ppo_ns3_online')

            print('Launching simulation')
            # View the model performance in live simulation
            while True:
                print("Start iteration: ", currIt)
                obs = env.reset()
                reward = 0
                done = False
                info = None
                print("Step: ", stepIdx)
                print("-- obs: ", obs)

                while True:
                    stepIdx += 1
                    action, _states = model.predict(obs)
                    print("Predicted action: ", action, type(action))

                    print("Step: ", stepIdx)
                    obs, reward, done, _ = env.step(action)

                    print(f'Obs: {obs}, Reward: {reward}, Done: {done}')

        except KeyboardInterrupt:
            env.close()
            print("Ctrl-C -> Exit")

        finally:
            env.close()
            print("Done")
    elif sys.argv[1] in ['train'] and sys.argv[2] in ['offline']:
        # Train using the RSU custom gym environment
        # Create environment
        env = gym.make("rsu-v0")

        # Use the stable-baseline PPO policy to train
        model = PPO2('MlpPolicy',
                     env,
                     verbose=1,
                     ent_coef=0.0,
                     lam=0.94,
                     gamma=0.99,
                     tensorboard_log='rsu_agents/ppo_offline_tensorboard/')

        # Train the agent
        print("Beginning model training")
        model.learn(total_timesteps=int(2e5))
        print("** Done training the model **")

        # Save the agent
        model.save('rsu_agents/ppo_offline_weights')

        # deleting it just to make sure we can load successfully again
        del model

        # Re-load the trained PPO algorithm with
        # parameters saved as 'ppo_rsu'
        model = PPO2.load('rsu_agents/ppo_offline_weights')

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
