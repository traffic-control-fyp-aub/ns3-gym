import pytest
import gym
import numpy as np
from gym_rsu.envs.rsu_env import RSUEnv

"""
    This is a test suite for the purpose of
    verifying the functionalities present in
    the custom Gym RSU environment.
"""


@pytest.fixture
def rsu_env():
    """
        Test fixture that creates a default RSUEnv
        object every time it is called.

        Return(s):
        ----------
            Returns a blank slate RSU Environment
    """
    return RSUEnv()


@pytest.mark.parametrize("obs_space, action_space, reward", [(gym.spaces.Box(np.array([0, 0]),
                                                                             np.array([2, 3.5]),
                                                                             dtype=np.float16),
                                                              gym.spaces.Box(np.array([-1]),
                                                                             np.array([1]),
                                                                             dtype=np.float16),
                                                              0)])
def test_env_init(rsu_env, obs_space, action_space, reward):
    """
        Test that the RSUEnv is initialized properly.

        Parameter(s):
        -------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
        obs_space: type(gym.spaces.Box)
            Observation space specific to the RSUEnv
        action_space: type(gym.spaces.Box)
            Action space specific to the RSUEnv
        reward: type(Float)
            Reward collected by the agent from the environment
            whenever an environment action is taken.
    """
    # check the observation space
    assert rsu_env.observation_space == obs_space

    # check the action space
    assert rsu_env.action_space == action_space

    # check the empty reward
    assert rsu_env.reward == reward


def test_poisson_sampling(rsu_env, q_value):
    """
        Test the Poisson sampling utility function
        in the RSU gym environment.

        Parameter(s):
        ------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
        q_value: type(Float)
            Flow value
    """
    pass


def test_exponential_sampling(rsu_env, q_value):
    """
        Test the Exponential sampling utility function
        in the RSU gym environment.

        Parameter(s):
        ------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
        q_value: type(Float)
            Flow value
    """
    pass


def test_take_action(rsu_env, action):
    """
        Test the take action utility function
        in the RSUEnv.

        Parameter(s):
        -------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
        action: type(Numpy Array)
            Array of de/acceleration values for the vehicles
            present in the RSUEnv
    """
    pass


def test_next_observation(rsu_env):
    """
        Test the next observation
        utility function in the RSUEnv

        Parameter(s):
        -------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
    """
    pass


def test_env_reset(rsu_env):
    """
        Test the environmental reset function
        of the RSUEnv.

        Parameter(s):
        -------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
    """
    pass


def test_step_func(rsu_env, action):
    """
        Test the step function in the RSUEnv.

        Parameter(s):
        -------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
        action: type(Numpy Array)
            Array of de/acceleration values for the vehicles
            present in the RSUEnv
    """
    pass
