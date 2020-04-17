from agent_utils.model_setup import model_setup


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

from stable_baselines import PPO2, SAC, TD3
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv


# Dummy variable that we use in order to get rid of a bug we
# were facing in regards to training in a vectorized environment
# which is necessary for some algorithms such as PPO
ns3_obj = None

# This variable is used to help facilitate the process of setting up
# the agent based on the CLI input specified by the user
model_online = None


def make_ns3_env():
    """
        This function is created as a bug fix to train
        our PPO agent which requires that all its environments
        be wrapped in a vectorized environment layer.
    """
    return ns3_obj


def error_on_specification():
    # User did not specify the file properly
    print("Please specify one of the following: [ test | train ]"
          " and if you specified to train then [ --online | --offline ]")


def test_algorithm():
    # Load the previously trained agent parameters and start
    # running the traffic simulation
    # Creating the ns3 environment that will act as a link
    # between our agent and the live simulation
    env = ns3env.Ns3Env(port=5555,
                        stepTime=0.5,
                        startSim=0,
                        simSeed=12,
                        debug=False)

    ob_space = env.observation_space
    ac_space = env.action_space

    print("Observation Space: ", ob_space, ob_space.dtype)
    print("Action Space: ", ac_space, ac_space.dtype)

    stepIdx, currIt = 0, 0

    try:
        model = TD3.load(f'rsu_agents/single_lane_highway_agents/optimized_interval/'
                         f'TD3_ns3_single_lane_highway_cars=25_optimized')

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

    finally:
        env.close()
        print("Done")


def train_agent_speed_online():
    global speed_online_model

    # Get the name of the RL policy algorithm
    rl_agent_name = input("\nEnter the name of the RL algorithm to use: ")

    # Get the name of the traffic scenario
    traffic_scenario_name = input("\nEnter the name of the traffic scenario: ")

    # Do they want to specify policy kwargs ?
    specify_policy_kwargs = input("\nEnter any policy kwargs, otherwise type in `pass`: ")

    # Dictionary to store the user specified model parameters
    params_dict = dict()

    # Check whether the user has entered model paramters via the CLI
    entered_cli = False if specify_policy_kwargs.lower() == 'pass' else True

    if entered_cli:
        print('Adding user specified CLI parameters in the following format: XX=YY')
        print('Enter the word `Done` once you are finished:\n')
        # Get the rest of the parameters specified by the user in the CLI
        # only if the user actually specified any. Otherwise no point in doing
        # so and use the default parameters
        params = input()
        while params.lower() != 'done':
            # Split the string into the variable name and the variable value
            params = params.split("=")

            # Add a new entry into the dictionary with the key being the variable
            # name and the value being the variable value
            params_dict[params[0]] = params[1]

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

    try:
        print('Setting up the model')

        if entered_cli:
            # Case where user has specified some CLI arguments for the agent

            # --------------------------------------------------------------
            # Use the part below when looking to perform base learning
            # --------------------------------------------------------------
            speed_online_model = model_setup(str(rl_agent_name),
                                             env,
                                             'MlpPolicy',
                                             lr=float(params_dict["lr"]),
                                             bf=int(params_dict["bf"]),
                                             bch=int(params_dict["bch"]),
                                             ent=str(params_dict["ent"]),
                                             tf=int(params_dict["tf"]),
                                             grad=int(params_dict["grad"]),
                                             lst=int(params_dict["lst"]),
                                             v=int(params_dict["v"]))

                # --------------------------------------------------------------
                # Use the part below when looking to perform continuous learning
                # --------------------------------------------------------------
                # print(f'Loading {str(argumentList[agent_index])} agent with cars={params_dict["cars"]}')
                # model_online = PPO2.load(f'rsu_agents/{traffic_scenario_name}_agents/optimized_interval/'
                #                          f'{str(argumentList[agent_index])}_ns3_{traffic_scenario_name}_cars='
                #                          f'{params_dict["cars"]}_optimized')
                #
                # # Setting the environment to allow the loaded agent to train
                # model_online.set_env(env=env)
        else:
            print(f'Setting up default {str(rl_agent_name)} parameters')
            # Otherwise just set up the model and use the default values
            speed_online_model = model_setup(str(rl_agent_name), env, 'MlpPolicy')

        print('Training model')
        # Start the learning process on the ns3 + SUMO environment
        speed_online_model.learn(total_timesteps=30000)
        print(' ** Done Training ** ')
    except KeyboardInterrupt:
        speed_online_model.save(f'rsu_agents/speed_control_agents/{traffic_scenario_name}_agents/'
                          f'optimized_interval/{rl_agent_name.capitalize()}_cars={str(ac_space.shape)[1:3]}_optimized')
        env.close()

    finally:
        env.close()


def train_agent_speed_offline():
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
    model.save('rsu_agents/ppo_ns3_offline')

    # deleting it just to make sure we can load successfully again
    del model

    # Re-load the trained PPO algorithm with
    # parameters saved as 'ppo_rsu'
    model = PPO2.load('rsu_agents/ppo_ns3_offline')

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


def train_agent_lane_online():
    global lane_online_model

    # Get the name of the RL policy algorithm
    rl_agent_name = input("\nEnter the name of the RL algorithm to use: ")

    if rl_agent_name.capitalize() != 'DQN':
        raise Exception("We currently only support DQN for lane changing.")

    # Get the name of the traffic scenario
    traffic_scenario_name = input("\nEnter the name of the traffic scenario: ")

    # Train using the ns3 SUMO environment
    # Creating the ns3 environment that will act as a link
    # between our agent and the live simulation
    ns3_obj = ns3env.Ns3Env(port=5555,  # FIXME - Are we still going to connect to port 5555?
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

    try:
        print('Setting up the model')

        if entered_cli:
            # Case where user has specified some CLI arguments for the agent

            # --------------------------------------------------------------
            # Use the part below when looking to perform base learning
            # --------------------------------------------------------------
            lane_online_model = model_setup(str(rl_agent_name).capitalize(),
                                            env,
                                            'MlpPolicy')

        else:
            print(f'Setting up default {str(rl_agent_name)} parameters')
            # Otherwise just set up the model and use the default values
            lane_online_model = model_setup(str(rl_agent_name).capitalize(), env, 'MlpPolicy')

        print('Training model')
        # Start the learning process on the ns3 + SUMO environment
        lane_online_model.learn(total_timesteps=30000)
        print(' ** Done Training ** ')
    except KeyboardInterrupt:
        lane_online_model.save(f'rsu_agents/lane_control_agents/{traffic_scenario_name}_agents/'
                               f'optimized_interval/{rl_agent_name.capitalize()}_cars='
                               f'{str(ac_space.shape)[1:3]}_optimized')
        env.close()

    finally:
        env.close()
