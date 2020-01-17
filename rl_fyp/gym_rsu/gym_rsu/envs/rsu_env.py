import gym
import numpy as np
import random
import pandas as pd


"""
    Custom OpenAI Gym environment from the perspective
    of the road side unit (RSU).
"""
PATH_TO_DATA_FRAME = "/home/rayyan/Desktop/FYP/repos/ns3-gym/rl_fyp/training_data/training_data_frame.csv"
MAX_HEADWAY_TIME = 2
MAX_VELOCITY_VALUE = 3.5
MAX_STEPS = 5
ALPHA = 0.1  # gain used to diminish the magnitude of the penalty
DESIRED_VELOCITY = 3  # desired system wide target (average) velocity
NUMBER_OF_VEHICLES = 3  # number of vehicles present in the environment
TOTAL_SECONDS_OF_INTEREST = 60*60  # 60 seconds/minute * 60 minutes


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

            self.reward: type(Float)
                This is the cumulative reward range allowed during
                an episode of interaction between the agent
                and the gym environment.

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
        self.observation_space = gym.spaces.Box(np.array([0, 0]),
                                                np.array([MAX_HEADWAY_TIME, MAX_VELOCITY_VALUE]),
                                                dtype=np.float16)

        self.action_space = gym.spaces.Box(np.array([-1]),
                                           np.array([1]),
                                           dtype=np.float16)
        self.reward = 0

        try:
            self.df = pd.read_csv(PATH_TO_DATA_FRAME)
        except FileNotFoundError:
            print(f'The provided path to training data frame does not exist: {PATH_TO_DATA_FRAME}')

        self.current_step = 0

    def step(self, action):
        """
            Step function to be taken on the environment.

            Parameter(s):
            -------------
            action: type(gym.space.Box)
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
        if not isinstance(action, np.ndarray):
            raise Exception(f'Action must be of type Numpy Array instead is of type {type(action)}')
        self._take_action(action)
        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Headway'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        # create a dataframe of values all set to desired velocity but same size as Velocity column
        desired_velocity = {'V_Desired': [DESIRED_VELOCITY]}
        desired_velocity_dataframe = pd.DataFrame(desired_velocity, columns=['V_Desired'])

        # create a dataframe of values all set to desired headway but same size as Headway column
        desired_headway = {'Head_Desired': [MAX_HEADWAY_TIME]}
        desired_headway_dataframe = pd.DataFrame(desired_headway, columns=['Head_Desired'])

        for ii in self.df['Velocity'].__len__() - 1:
            desired_velocity_dataframe.append(DESIRED_VELOCITY)
            desired_headway_dataframe.append(MAX_HEADWAY_TIME)

        temp = []
        for jj in self.df['Headway'].__len__():
            temp.append(max(desired_headway_dataframe.loc[jj] - self.df['Headway'].loc[jj], 0))

        list_of_maximums = np.asarray(temp)

        #   Based on a modified version of the reward function present in the following paper:
        #   Dissipating stop-and-go waves in closed and open networks via deep reinforcement learning
        #   By A. Kreidieh
        #
        #   Below is mathematical form of our reward function:
        #   ||v_desired|| - (1-alpha)(||v_desired - v_i(t)||) - (alpha)( summation(max(h_max - h_i(t), 0)) )
        self.reward = abs(DESIRED_VELOCITY)\
            - (1-ALPHA)*abs((np.sum(np.subtract(desired_velocity_dataframe.values,
                            self.df['Velocity'].values)))) - ALPHA*(sum(list_of_maximums))

        self.reward *= delay_modifier
        done = False  # FIXME - define ending condition

        obs = self._next_observation()

        return obs, self.reward, done, {}

    def reset(self):
        """
            Reset function that sets the environment back
            to it's initial settings.
        """
        # Set the current step to a random point within frame
        self.current_step = random.randint(0, len(self.df.loc[:, 'Headway'].values) - 6)

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
            at each time step.
        """
        print(f'Step: {self.current_step}')
        print(f'Velocity of all vehicles: {self.df["Velocity"]}')
        print(f'********************')

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

                * Note:
                -------
                All values are scaled between 0 and 1
        """
        obs = np.array([
            self.df.loc[self.current_step:
                        self.current_step + 5, 'Headway'].values
            / MAX_HEADWAY_TIME,
            self.df.loc[self.current_step:
                        self.current_step + 5, 'Velocity'].values
            / MAX_VELOCITY_VALUE])

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
        if action.__len__() < NUMBER_OF_VEHICLES:
            raise Exception(f"Size of action list does not match number of vehicles: {NUMBER_OF_VEHICLES}")
        else:
            for index, elem in self.df['Velocity'].__len__():
                self.df['Velocity'].loc[index] = elem + action[index] if action[index] >= 0 else elem - action[index]

            # Knowing the new set of velocities for the vehicles we now need to compute the
            # new set of headways since the previously recorded ones are useless. The following
            # proposed solution methodology is what we follow through with:
            #     1) Derive the new system average velocity over all the vehicles after applying the action.
            #     2) Derive the average number of vehicles arriving per hour.
            #     3) If the value is < 2000 vehicles/hr then the headway time follows a poisson distribution
            #     4) Else if the value is > 2000 vehicles/hr then the headway times follows the exponential distribution
            #     5) Sample from the chosen headway distribution and update the headway times.
            average_velocity = sum(self.df['Velocity']) / self.df['Velocity'].__len__()
            q_flow_value = average_velocity * TOTAL_SECONDS_OF_INTEREST

            if q_flow_value < 2000:
                # poisson distribution
                self._sample_poisson_value(q_flow_value)
            elif q_flow_value >= 2000:
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
        for ii in self.df['Headway'].__len__():
            self.df['Headway'].loc[ii] = np.random.poisson(q)

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
        for ii in self.df['Headway'].__len__():
            self.df['Headway'].loc[ii] = np.random.exponential(q)
