import gym
from gym.utils import seeding

import numpy as np
import pandas as pd
from beautifultable import BeautifulTable

import math
import random

"""
    Custom OpenAI Gym environment from the perspective
    of the road side unit (RSU).
"""
DIRECT_PATH_TO_DATA_FRAME = "/home/rayyan/Desktop/FYP/repos/ns3-gym/rl_fyp/training_data/training_data_frame.csv"
PATH_TO_DATA_FRAME = "rl_fyp/training_data/training_data_frame.csv"
MAX_HEADWAY_TIME = 2  # maximum allowed headway time for vehicles in seconds
MAX_VELOCITY_VALUE = 3.5  # maximum allowed velocity for vehicles in meters per second
ALPHA = 0.1  # gain used to diminish the magnitude of the penalty
DESIRED_VELOCITY = 3  # desired system wide target (average) velocity
NUMBER_OF_VEHICLES = 4  # number of vehicles present in the environment
TOTAL_SECONDS_OF_INTEREST = 60*15  # 60 seconds/minute * 15 minutes
EPSILON_THRESHOLD = math.pow(10, -5)  # threshold used to check if reward is advancing or not
CIRCUIT_LENGTH = 1500  # length of the traffic circuit environment
FLOW_WINDOW_CONSTANT = 15  # flow volume within the window frame of 15 minutes
TRAFFIC_FLOW_THRESHOLD = 1.4  # Flow Q-value threshold (reported commonly in traffic literature)
MEAN_VELOCITY = 1.75  # value to center normal distribution velocity sampling
MEAN_HEADWAY = 1.5  # value to center normal distribution headway sampling
SIGMA = 0.1  # standard deviation for normal distribution velocity sampling
BETA = 0.99  # constant to be used in the delay modifier calculation


class RSUEnv(gym.Env):
    """
        Custom environment that follows the Gym
        interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
            Initialization constructor for the
            custom environment.

            Attributes:
            -----------
            self.action_space: type(gym.space.Box)
                This is the action space allowable in our
                environment. Specifically, it is the normalized
                set of continuous numbers between -1 and +1 where
                {-1 = complete deceleration, +1 = complete acceleration}

            self.observation_space: type(gym.space.Box)
                This is the observation space allowable in our
                environment. Specifically, it is the set of
                velocities and headway times being broadcasted
                by the vehicles. The following values are observed:
                    - Headway Time in seconds: [0, 2]
                    - Vehicle Velocity in m/s: [0, 3.5]

            self.current_reward: type(Float)
                This is the cumulative reward range allowed during
                an episode of interaction between the agent
                and the gym environment.

            self.old_reward: type(Float)
                This is the reward that was observed 10 time steps ago
                in the environment. It is used in the step function to
                tell if there is any advancement in the training and
                to determine whether or not to end the training episode.

            self.df: type(dataframe)
                Dataframe that contains the information
                to use while training on this environment.

            self.current_step: type(Int)
                Only used in the reset method. Within the reset
                method we give it a random value to point to within
                the data frame because this gives our agent a more
                unique experience from the dame data set.

            self.first_run: type(bool)
                Use this parameter to see if we are running the environment
                for the first time or not in order to decide whether the agent
                samples randomly or submit a list of actions.
        """
        super(RSUEnv, self).__init__()

        # Initializing my observation space to be a vector of length 2*NUMBER_OF_VEHICLES
        # where the first "NUMBER_OF_VEHICLES" observations correspond to the headway times
        # and the remaining "NUMBER_OF_VEHICLES" observations correspond to the vehicle velocities.
        self.observation_space = gym.spaces.Box(low=0,
                                                high=MAX_VELOCITY_VALUE,
                                                shape=(2*NUMBER_OF_VEHICLES,),
                                                dtype=np.float32)

        # Initializing my action space to be a vector of length NUMBER_OF_VEHICLES
        # which consists of a continuous interval from -1 to +1
        self.action_space = gym.spaces.Box(low=-1,
                                           high=1,
                                           shape=(NUMBER_OF_VEHICLES,),
                                           dtype=np.float32)

        # Zero-ing out the reward values
        self.current_reward, self.old_reward = 0, 0

        # Setting current time step to zero
        self.current_step = 0

        # Specify that this is the first time we run the environment
        self.first_run = True

        # Initializing random seed of RSU environment
        self._seed()

        # Opening data frame that contains environment related data
        try:
            self.df = pd.read_csv(PATH_TO_DATA_FRAME)
        except FileNotFoundError:
            # Re-try importing the CSV file because sometimes the
            # relative import does not find the CSV file.
            print(f'The provided path to training data frame does not exist: {PATH_TO_DATA_FRAME}')
            print(f'Switching to absolute path instead: {DIRECT_PATH_TO_DATA_FRAME}')
            self.df = pd.read_csv(DIRECT_PATH_TO_DATA_FRAME)

    def step(self, action=None):
        """
            Step function to be taken on the environment.

            Parameter(s):
            -------------
            action: type(ndarray || list || object)
                    The action to be taken by the agent that
                    will affect the state of the environment.

            Return(s):
            ----------
            obs: Object
                An environment specific object representing
                the agent's observation of the environment.
            reward: Float
                    Amount of reward achieved by the previous action.
            done:   Boolean
                    Whether it's time to reset the environment again.
            info:   Dictionary
                    Diagnostic information useful for debugging.
        """
        # Take a single step in time and enforce its effects on the RSU environment
        if self.first_run:
            # It is no longer the first time we run a step
            # in the RSU environment
            self.first_run = False

            # If the current time step is still zero then
            # think about just running a randomly sampled
            # list of actions
            self._take_action(self._action_to_list(self.action_space.sample()))
        else:
            # Convert the action to type list
            action = self._action_to_list(action)

            # Take environment effect of action list
            self._take_action(action)

        # Advance the current step of the environment by 1
        self.current_step += 1

        desired_velocity = np.asarray([])
        for _ in range(len(self.df['Velocity'].values)):
            desired_velocity = np.append(desired_velocity, DESIRED_VELOCITY)

        desired_headway = np.asarray([])
        for _ in range(len(self.df['Headway'].values)):
            desired_headway = np.append(desired_headway, MAX_HEADWAY_TIME)

        desired_dataframe = pd.DataFrame({'H_desired': desired_headway, 'V_desired': desired_velocity})

        temp = []
        for index, _ in self.df.iterrows():
            temp.append(max(desired_dataframe.loc[index, 'H_desired'] - self.df.loc[index, 'Headway'], 0))

        list_of_maximums = np.asarray(temp)

        #   Based on a modified version of the reward function present in the following paper:
        #   Dissipating stop-and-go waves in closed and open networks via deep reinforcement learning
        #   By A. Kreidieh
        #
        #   Below is mathematical form of our reward function:
        #   ||v_desired|| - ( sum(||v_desired - v_i(t)|| ))/N - (alpha)( summation(max(h_max - h_i(t), 0)) )
        #
        #   where:
        #       - v_desired is the desired velocity of the system calculated by the road side unit
        #       - v_i(t) is the velocity of vehicle "i" at time "t"
        #       - h_i(t) is the headway time observed by vehicle "i" at time "t"
        #       - N is the total number of vehicles in the environment
        self.current_reward = abs(MAX_VELOCITY_VALUE)\
            - abs((np.sum(np.subtract(desired_dataframe['V_desired'].values,
                                      self.df['Velocity'].values))))/(len(self.df['Velocity'].values))\
            - ALPHA*(sum(list_of_maximums))

        #   Multiply by a delay modifier in order to encourage exploration in the long
        #   term and not just settle with a local maximum (i.e.: prefer long term to
        #   short term planning)
        reward = self.current_reward * BETA

        if self.current_step % len(self.df['Headway'].values) == 0:
            self.old_reward = self.current_reward

        #   Condition that would trigger the end of an episode.
        #   If the Mean Squared Error between the current reward and the reward
        #   that was observed 10 time steps ago did not change beyond a certain
        #   threshold then the training is done.
        done = True if math.pow(abs(self.old_reward - self.current_reward), 2) < EPSILON_THRESHOLD else False

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        """
            Reset function that sets the environment back
            to it's initial settings.
        """
        # Set the current step to 0
        self.current_step = 0

        velocities = np.asarray([])
        for _ in range(4):
            # Generate new samples of velocities from a normal distribution
            # centered around the mean velocity with standard deviation sigma.
            velocities = np.append(velocities, round(abs(np.random.normal(MEAN_VELOCITY, SIGMA)), 2))

        headways = np.asarray([])
        for _ in range(4):
            # Generate new samples of headways from a normal distribution
            # centered around the mean headway with standard deviation sigma.
            headways = np.append(headways, round(abs(np.random.normal(MEAN_HEADWAY, SIGMA)), 2))

        # Two dimentional dataframe consisting of the sampled headways and velocities
        training_dataframe = pd.DataFrame({'Headway': headways, 'Velocity': velocities})

        # Export the dataframe containing all the training data
        # as a CSV file located at PATH_TO_CSV
        try:
            _ = training_dataframe.to_csv(PATH_TO_DATA_FRAME)
        except FileNotFoundError:
            _ = training_dataframe.to_csv(DIRECT_PATH_TO_DATA_FRAME)

        # Return a new observation from the RSU environment
        obs = self._next_observation()

        return obs

    def render(self, mode='human'):
        """
            Function that renders the environment to the
            user.

            *Note:
            ------
            It is up to the developer to choose their own
            definition of the render function. It is not
            necessary that the function output some visual
            graphics to the screen.

            Here we chose to print the velocities of all the vehicles
            at each time step using the BeautifulTable package.
        """
        print(f' - Step: {self.current_step}')
        table = BeautifulTable()
        table.column_headers = ["Headway", "Velocity"]
        for index, row in self.df.iterrows():
            table.append_row([self.df.at[index, 'Headway'], self.df.at[index, 'Velocity']])
        print(table)

    def close(self):
        """
            This function ensures that the environment
            is closed properly.
        """
        self.close()

    def _seed(self, seed=None):
        """
            Utility function that helps with setting
            the seed for random sampling.

            Parameter(s):
            -------------
            seed: type(Float)
                User set seed value for seeding function.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _next_observation(self):
        """
            Helper function that returns the next
            observation in the environment.

            Return(s):
            ----------
            obs: type(list)
                Next observation in the environment.
                Of the form: (h_t+1, v_t+1) where:
                    - h_t+1 = next time step headway
                    - v_t+1 = next time step velocity
                Total length of the observation vector
                is 2N where:
                    - N = number of vehicles on the circuit
        """
        o = np.asarray([])
        for index, _ in self.df.iterrows():
            # Appending all the headway times for the next time step
            o = np.append(o, self.df.loc[index, 'Headway'])

        for index, _ in self.df.iterrows():
            # Appending all the velocity values of the next time step
            o = np.append(o, self.df.loc[index, 'Velocity'])

        # Converting the numpy array to type list
        obs = o.tolist()

        return obs

    def _take_action(self, action):
        """
            Take the action provided by the agent/model
            and physically perform it on the environment.

            Specifically what happens is that based on the
            received action the RSU then modifies the following
            time step's velocities. These future time step velocity
            values can be accessed from the recorded data in the
            data frame specified within the RSU environment.
            This can be viewed as the equivalent of training from
            scratch without any previously recorded data.

            * Note:
            -------
            This is not to be confused with the action function within the
            agent that is responsible for actually taking the decision of speeding
            up or slowing down and by a certain amount.

            Parameter(s):
            -------------
            action: type(Numpy Array(dtype=Float))
                Since this is an environment from the perspective of the RSU
                then the action received will be a individual de/acceleration(s)
                to be performed on the vehicle(s). The length of this vector
                is equal to the number of vehicles in the environment.
        """
        if len(action) != NUMBER_OF_VEHICLES:
            raise Exception(f"Size of action list does not match number of vehicles: {len(action)}."
                            f" Here is that action: {action}")
        else:
            for index, _ in self.df.iterrows():
                self.df.loc[index, 'Velocity'] = (self.df.loc[index, 'Velocity'] + action[index])\
                                                 % MAX_VELOCITY_VALUE

                # Adjust the headway time of the current vehicle of focus
                self._adjust_relative_headway(index, action[index])

    def _adjust_relative_headway(self, index, v_delta):
        """
            Utility function that proportionally decreases
            the headway value of a vehicle of focus.

            If the submitted action requires that a vehicle speeds up
            then it would be logical that the headway time decreases
            assuming the vehicle in front of it does not change speed.
            However, if the submitted action requires that a vehicle
            slows down then it would be logical that the headway time
            increases assuming the vehicle in front of it does not
            change speed.

            Parameter(s):
            -------------
            index: type(int)
                Index of the vehicle of focus in the dataframe.
            v_delta: type(float)
                Speed change instructed by the RSU.
        """
        if v_delta > 0:
            # RSU is telling the vehicle of focus to speed up
            # therefore the headway time must decrease
            # TODO - maybe make these values absolute?
            self.df.loc[index, 'Headway'] = (self.df.loc[index, 'Headway'] - (v_delta*SIGMA)) % MAX_HEADWAY_TIME
        elif v_delta < 0:
            # RSU is telling the vehicle of focus to slow down
            # therefore the headway time must increase
            self.df.loc[index, 'Headway'] = (self.df.loc[index, 'Headway'] + abs(v_delta*SIGMA)) % MAX_HEADWAY_TIME
        else:
            # RSU is telling the vehicle of focus to maintain speed
            # therefore the headway time must stay the same
            pass

    def _action_to_list(self, action):
        """
            This function is a utility function that
            converts the action vector into a list.

            Parameter(s):
            -------------
            action: type(ndarray || list || Object)
                Action vector of length NUMBER_OF_VEHICLES
                to be applied on the RSUEnv.
        """
        # Converting the values to be between -1 and 1
        for index in range(len(action)):
            temp = abs(action[index]) % 1
            if action[index] < 0:
                action[index] = round(-temp, 2)
            else:
                action[index] = round(temp, 2)
        if isinstance(action, np.ndarray):
            return action.tolist()
        if not isinstance(action, list):
            return [action]
        return action
