"""
    Usage:
    ------
    >> python3 script.py [ test | train ] [ online | offline ] [ algorithm name ] [scenario_name] [ algorithm_params ]
        + test: Load and test the performance of a previously trained algorithm into
            `   the ns3-SUMO simulator.
        + train: Train an agent from scratch either directly on the ns3-SUMO simulator
                 or through an offline custom openAI gym environment called RSUEnv

        * if you specified the 'test' parameter then you do no need to enter the choice of
          offline or online however you do still need to specify the algorithm name because
          we use that to load in the saved agent directly from the command line.

        * if you specified the 'train' parameter then you need to fill in all the remaining
          parameters [ online | offline ] and the algorithm you which to train with.

        * Here is a list of algorithms we already support:
            - PPO2

    *Note that model naming convention whenever training and then saving will
    always be:
                    rsu_agents/[scenario_name]_agents/[algorithm]_ns3_[scenario_name]_[num_of_cars]

    This will later be used to help facilitate testing the agents directly from
    the CLI instead of having to edit this script file every time.

    e.g.: (training)
    -----
    - PPO2
    >> python3 script.py train online PPO2 scenario=square lr=2.5e-4 v=1 ent=0.0 lbd=0.95 g=0.99

    - SAC ( we automatically fetch the algorithm's default parameters )
    >> python3 script.py train online SAC scenario=square

    e.g.: (testing)
    >> python3 script.py test scenario=square cars=10

"""
import sys
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

# Collect the list of command line arguments passed
argumentList = sys.argv


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


if argumentList.__len__() == 1:
    # User did not specify the file properly
    print("Please specify one of the following: [ test | train ]"
          " and if you specified to train then [ --online | --offline ]")
    exit(0)
elif argumentList.__len__() is 4:
    if sys.argv[1] in ['test']:

        # Collect from the CLI the name of the traffic scenario
        scenario_name = sys.argv[2].split("=")[1]

        # Collect from the CLI the number of cars that the agent was trained on
        num_of_vehicles = sys.argv[3].split("=")[1]

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

            # model = PPO2.load(f'rsu_agents/{scenario_name}_agents/'
            #                   f'PPO2_ns3_online_{scenario_name}_cars={num_of_vehicles}')

            # model = PPO2.load((f'rsu_agents/square_agents/PPO2_algorithm/'
            #                   f'PPO2_ns3_'
            #                   f'square_cars=25'))

            # model = SAC.load((f'rsu_agents/square_agents/SAC_algorithm/'
            #                   f'SAC_ns3_'
            #                   f'square_cars=25'))

            model = TD3.load((f'rsu_agents/square_agents/TD3_algorithm/'
                              f'TD3_ns3_'
                              f'square_cars=25'))

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
elif argumentList.__len__() >= 5:
    if sys.argv[1] in ['train'] and sys.argv[2] in ['online']:

        # Find the index of the agent name parameter
        agent_index = argumentList.index("online") + 1

        # Find the index of the traffic scenario specified
        traffic_scenario_index = agent_index + 1

        # Extracting the name of the traffic scenario to use for saving
        # the name of the agent
        traffic_scenario_name = argumentList[traffic_scenario_index].split("=")
        traffic_scenario_name = traffic_scenario_name[1]

        # Temp list of user specified parameters through the CLI
        params = None

        # Dictionary to store the user specified model parameters
        params_dict = dict()

        # Check whether the user has entered model paramters via the CLI
        entered_cli = False

        if agent_index < argumentList.__len__() - 1:
            print('Adding user specified CLI parameters')
            entered_cli = True
            # Get the rest of the parameters specified by the user in the CLI
            # only if the user actually specified any. Otherwise no point in doing
            # so and use the default parameters
            for list_index in range(traffic_scenario_index+1, argumentList.__len__()):
                # Split the string into the variable name and the variable value
                params = argumentList[list_index].split("=")

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

        stepIdx, currIt = 0, 0

        try:
            print('Setting up the model')

            if entered_cli:
                # Case where user has specified some CLI arguments for the agent
                model_online = model_setup(str(argumentList[agent_index]),
                                           env,
                                           'MlpPolicy',
                                           lr=float(params_dict["lr"]),
                                           nsteps=int(params_dict["nsteps"]),
                                           nbtch=int(params_dict["nbtch"]),
                                           lbd=float(params_dict["lbd"]),
                                           g=float(params_dict["g"]),
                                           nep=int(params_dict["nep"]),
                                           ent=float(params_dict["ent"]),
                                           cl=float(params_dict["cl"]),
                                           v=int(params_dict["v"]))

                # print(f'Loading {str(argumentList[agent_index])} agent with cars={params_dict["cars"]}')
                # model_online = PPO2.load(f'rsu_agents/{traffic_scenario_name}_agents/base_learning_no_traffic_light/'
                #                         f'{str(argumentList[agent_index])}_algorithm/{str(argumentList[agent_index])}'
                #                         f'_ns3_{traffic_scenario_name}_cars={params_dict["cars"]}')

                # Setting the environment to allow the loaded agent to train
                model_online.set_env(env=env)
            else:
                print(f'Setting up default {str(argumentList[agent_index])} parameters')
                # Otherwise just set up the model and use the default values
                model_online = model_setup(str(argumentList[agent_index]), env, 'MlpPolicy')

            print('Training model')
            # Start the learning process on the ns3 + SUMO environment
            model_online.learn(total_timesteps=30000)   # int(128*60000)) < -- PPO2
            print(' ** Done Training ** ')
        except KeyboardInterrupt:
            model_online.save(f'rsu_agents/{traffic_scenario_name}_agents/continuous_learning_traffic_light/'
                              f'{str(argumentList[agent_index])}_cl/'
                              f'{str(argumentList[agent_index])}_ns3_'
                              f'{traffic_scenario_name}_cars={str(ac_space.shape)[1:3]}_CL')
            env.close()
            print("Ctrl-C -> Exit")

        finally:
            env.close()
            print("Environment closed")
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
