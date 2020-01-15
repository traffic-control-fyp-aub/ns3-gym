import gym
import numpy as np
import random
import pandas as pd


"""
    Custom OpenAI Gym environment from the perspective
    of the road side unit (RSU).
"""
MAX_REWARD_RANGE = 1  # Upper bound on reward range
PATH_TO_DATA_FRAME = "/path/to/data_frame"  # FIXME - set correct path
MAX_HEADWAY_TIME = 2
MAX_VELOCITY_VALUE = 3.5
MAX_STEPS = 5


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
                                                np.array([2, 3.5]),
                                                dtype=np.float16)

        self.action_space = gym.spaces.Box(np.array([-1]),
                                           np.array([1]),
                                           dtype=np.float16)

        self.reward_range = (0, MAX_REWARD_RANGE)

        self.df = pd.read_csv(PATH_TO_DATA_FRAME)

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
        self._take_action(action)
        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Headway'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = 0 * delay_modifier  # FIXME - set reward function
        done = reward <= 0  # FIXME - define ending condition

        obs = self._next_observation()

        return obs, reward, done, {}

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
        """
        print(f'Step: {self.current_step}')  # FIXME - decide on what to render

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
            _take_action()
        """
        pass
