import gym
import numpy as np
import random
import pandas as pd
import math
from beautifultable import BeautifulTable


"""
    Custom OpenAI Gym environment from the perspective
    of the road side unit (RSU).
"""
DIRECT_PATH_TO_DATA_FRAME = "/home/rayyan/Desktop/FYP/repos/ns3-gym/rl_fyp/training_data/training_data_frame.csv"
PATH_TO_DATA_FRAME = "rl_fyp/training_data/training_data_frame.csv"
MAX_HEADWAY_TIME = 2  # maximum allowed headway time for vehicles in seconds
MAX_VELOCITY_VALUE = 3.5  # maximum allowed velocity for vehicles in meters per second
MAX_STEPS = 5  # maximum time steps for training horizon
ALPHA = 0.1  # gain used to diminish the magnitude of the penalty
DESIRED_VELOCITY = 3  # desired system wide target (average) velocity
NUMBER_OF_VEHICLES = 4  # number of vehicles present in the environment
TOTAL_SECONDS_OF_INTEREST = 60*15  # 60 seconds/minute * 15 minutes
EPSILON_THRESHOLD = math.pow(10, -5)  # threshold used to check if reward is advancing or not
CIRCUIT_LENGTH = 1500  # length of the traffic circuit environment
FLOW_WINDOW_CONSTANT = 15  # flow volume within the window frame of 15 minutes
TRAFFIC_FLOW_THRESHOLD = 1.4  # Flow Q-value threshold (reported commonly in traffic literature)


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

            self.df: type(String)
                Path to the dataframe that contains the information
                to use while training on this environment.

            self.current_step: type(Int)
                Only used in the reset method. Within the reset
                method we give it a random value to point to within
                the data frame because this gives our agent a more
                unique experience from the dame data set.
        """
        super(RSUEnv, self).__init__()

        # Initializing my observation space to be of two dimensions:
        #     - First dimension: a continuous interval between 0 and MAX_HEADWAY_TIME
        #     - Second dimension: a continuous interval between 0 and MAX_VELOCITY_VALUE
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]),
                                                high=np.array([MAX_HEADWAY_TIME, MAX_VELOCITY_VALUE]),
                                                dtype=np.float16)

        # Initializing my action space to be a vector of length NUMBER_OF_VEHICLES
        # which consists of a continuous interval from -1 to +1
        self.action_space = gym.spaces.Box(low=-1,
                                           high=+1,
                                           shape=(NUMBER_OF_VEHICLES,),
                                           dtype=np.float16)

        self.current_reward, self.old_reward = 0, 0

        try:
            self.df = pd.read_csv(PATH_TO_DATA_FRAME)
        except FileNotFoundError:
            # Re-try importing the CSV file because sometimes the
            # relative import does not find the CSV file.
            self.df = pd.read_csv(DIRECT_PATH_TO_DATA_FRAME)
            print(f'The provided path to training data frame does not exist: {PATH_TO_DATA_FRAME}')

        self.current_step = 0

    def step(self, action=np.array([])):
        """
            Step function to be taken on the environment.

            Parameter(s):
            -------------
            action: type(Numpy Array)
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
        # Execute one time step within the environment
        if 'numpy' not in type(action).__module__:
            raise Exception(f'Action must be of type Numpy Array instead is of type {type(action)}')

        self._take_action(action)
        self.current_step = (self.current_step + 1) % len(self.df['Headway'].values)

        delay_modifier = (self.current_step / MAX_STEPS)

        desired_velocity = np.asarray([])
        for _ in range(len(self.df['Velocity'].values)):
            desired_velocity = np.append(desired_velocity, DESIRED_VELOCITY)

        desired_headway = np.asarray([])
        for _ in range(len(self.df['Headway'].values)):
            desired_headway = np.append(desired_headway, MAX_HEADWAY_TIME)

        desired_dataframe = pd.DataFrame({'H_desired': desired_headway, 'V_desired': desired_velocity})

        temp = []
        for jj in range(len(self.df['Headway'].values)):
            temp.append(max(desired_dataframe.loc[jj, 'H_desired'] - self.df['Headway'].loc[jj], 0))

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

        self.current_reward = abs(DESIRED_VELOCITY)\
            - abs((np.sum(np.subtract(desired_dataframe['V_desired'].values,
                                      self.df['Velocity'].values))))/(len(self.df['Velocity'].values))\
            - ALPHA*(sum(list_of_maximums))

        #   Multiply by a delay modifier in order to encourage exploration in the long
        #   term and not just settle with a local maximum (i.e.: prefer long term to
        #   short term planning)
        self.current_reward *= delay_modifier

        if self.current_step % 10 == 0:
            self.old_reward = self.current_reward

        #   Condition that would trigger the end of an episode.
        #   If the Mean Squared Error between the current reward and the reward
        #   that was observed 10 time steps ago did not change beyond a certain
        #   threshold then the training is done.
        done = True if math.pow(abs(self.old_reward - self.current_reward), 2) < EPSILON_THRESHOLD else False

        obs = self._next_observation()

        return obs, self.current_reward, done, {}

    def reset(self):
        """
            Reset function that sets the environment back
            to it's initial settings.
        """
        # Set the current step to a random point within frame
        self.current_step = random.randint(0, len(self.df.loc[:, 'Headway'].values))\
            % (len(self.df.loc[:, 'Headway'].values) - 2)

        self._next_observation()

    def render(self, mode='human', close=False):
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

    def _next_observation(self):
        """
            Helper function that returns the next
            observation in the environment.

            Return(s):
            ----------
            obs: type(Numpy Array)
                Next observation in the environment.
                Of the form: (h_t+1, v_t+1) where:
                    - h_t+1 = next time step headway
                    - v_t+1 = next time step velocity
                Total length of the observation vector
                is 2N where:
                    - N = number of vehicles on the circuit

                * Note:
                -------
                All values are scaled between 0 and 1
        """
        obs = np.asarray([])
        for index, _ in self.df.iterrows():
            obs = np.append(obs, self.df.loc[index, 'Headway'] / MAX_HEADWAY_TIME)

        for index, _ in self.df.iterrows():
            obs = np.append(obs, self.df.loc[index, 'Velocity'] / MAX_VELOCITY_VALUE)

        return obs

    def _take_action(self, action=np.array([])):
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
        if 'numpy' not in type(action).__module__:
            raise Exception(f'Action must be of type Numpy Array instead is of type {type(action)}')

        if len(action) != NUMBER_OF_VEHICLES:
            raise Exception(f"Size of action list does not match number of vehicles: {len(action)}."
                            f" Here is that action: {action}")
        else:
            for index, row in self.df.iterrows():
                self.df.loc[index, 'Velocity'] += action[index]

            # Knowing the new set of velocities for the vehicles we now need to compute the
            # new set of headways since the previously recorded ones are useless. The following
            # proposed solution methodology is what we follow through with:
            #     1) Derive the new system average velocity over all the vehicles after applying the action.
            #     2) Derive the average number of vehicles arriving per hour.
            #     3) If the value is < 2000 vehicles/hr then the headway time follows a poisson distribution
            #     4) Else if the value is > 2000 vehicles/hr then the headway times follows an exponential distribution
            #     5) Sample from the chosen headway distribution and update the headway times.

            # Average velocity of all vehicles in the traffic circuit environment
            average_velocity = sum(self.df['Velocity'].values) / len(self.df['Velocity'].values)

            # Total time it would take one vehicle on average to travel the entire traffic ciruict
            time_to_travel_circuit = CIRCUIT_LENGTH / average_velocity

            # Total amount of traffic volume in the circuit in a 15 minute time frame
            traffic_volume = TOTAL_SECONDS_OF_INTEREST / time_to_travel_circuit

            # Traffic Q-value flow rate
            q_flow_value = traffic_volume * FLOW_WINDOW_CONSTANT

            if q_flow_value < TRAFFIC_FLOW_THRESHOLD:
                # poisson distribution
                self._sample_poisson_value(q_flow_value)
            elif q_flow_value >= TRAFFIC_FLOW_THRESHOLD:
                # exponential distribution
                self._sample_exponential_value(q_flow_value)

    def _sample_poisson_value(self, q):
        """
            Draws samples from a Poisson Distribution
            and updates the previously recorded headway times
            accordingly.

            Parameter(s):
            -------------
            q: type(Float)
                Flow value
        """
        for index, row in self.df.iterrows():
            self.df.loc[index, 'Headway'] = np.random.poisson(q) % MAX_HEADWAY_TIME

    def _sample_exponential_value(self, q):
        """
            Draws samples from an Exponential Distribution
            centered around "q" and updates the previously
            recorded headway times accordingly.

            Parameter(s):
            -------------
            q: type(Float)
                Flow value
        """
        for index, row in self.df.iterrows():
            self.df.loc[index, 'Headway'] = np.random.exponential(q) % MAX_HEADWAY_TIME
