import pytest
import gym
import numpy as np
import math

from rl_fyp.gym_rsu.gym_rsu.envs.rsu_env import RSUEnv

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
    # Doing this only to get rid of an IDE glitch that does not
    # let me access the dataframe of the RSUEnv()
    if not isinstance(rsu_env, RSUEnv):
        raise Exception("Wrong environment")

    # check the observation space
    assert rsu_env.observation_space == obs_space

    # check the action space
    assert rsu_env.action_space == action_space

    # check the empty reward
    assert rsu_env.current_reward == reward


@pytest.mark.parametrize("q_value, epsilon, headway",
                         [(5, 5, 2)])
def test_poisson_sampling(rsu_env, q_value, epsilon, headway):
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
    # Doing this only to get rid of an IDE glitch that does not
    # let me access the dataframe of the RSUEnv()
    if not isinstance(rsu_env, RSUEnv):
        raise Exception("Wrong environment")

    rsu_env._sample_poisson_value(q_value)

    # Go through each element in the newly populated headway times
    # and make sure that they are correct. But since we are sampling
    # from a probability distribution we can not get the exact value
    # instead we can only check that the mean squared error between
    # what we got and what we would get if we sampled again is less
    # than a certain epsilon threshold.
    for index, row in rsu_env.df.iterrows():
        assert math.pow(abs(rsu_env.df.at[index, 'Headway'] - np.random.poisson(q_value)) % headway, 2) < epsilon


@pytest.mark.parametrize("q_value, epsilon, headway",
                         [(5, 5, 2)])
def test_exponential_sampling(rsu_env, q_value, epsilon, headway):
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
    # Doing this only to get rid of an IDE glitch that does not
    # let me access the dataframe of the RSUEnv()
    if not isinstance(rsu_env, RSUEnv):
        raise Exception("Wrong environment")

    rsu_env._sample_exponential_value(q_value)

    # Go through each element in the newly populated headway times
    # and make sure that they are correct. But since we are sampling
    # from a probability distribution we can not get the exact value
    # instead we can only check that the mean squared error between
    # what we got and what we would get if we sampled again is less
    # than a certain epsilon threshold.
    for index, row in rsu_env.df.iterrows():
        assert math.pow(abs(rsu_env.df.at[index, 'Headway'] - np.random.exponential(q_value)) % headway, 2) < epsilon


@pytest.mark.parametrize("next_headway, next_velocity, max_headway, max_velocity",
                         [(1, 2, 2, 3.5)])
def test_next_observation(rsu_env, next_headway, next_velocity, max_headway, max_velocity):
    """
        Test the next observation
        utility function in the RSUEnv

        Parameter(s):
        -------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
    """
    # Doing this only to get rid of an IDE glitch that does not
    # let me access the dataframe of the RSUEnv()
    if not isinstance(rsu_env, RSUEnv):
        raise Exception("Wrong environment")

    # Next time step headway value
    assert rsu_env._next_observation()[0] == next_headway / max_headway

    # Next time step velocity value
    assert rsu_env._next_observation()[1] == next_velocity / max_velocity


@pytest.mark.parametrize("next_headway, next_velocity, max_headway, max_velocity",
                         [(1, 2, 2, 3.5)])
def test_env_reset(rsu_env, next_headway, next_velocity, max_headway, max_velocity):
    """
        Test the environmental reset function
        of the RSUEnv.

        Parameter(s):
        -------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
    """
    # Doing this only to get rid of an IDE glitch that does not
    # let me access the dataframe of the RSUEnv()
    if not isinstance(rsu_env, RSUEnv):
        raise Exception("Wrong environment")

    # Reset the RSUEnv gym environment
    rsu_env.reset()

    # Get the newly set random time step
    new_time_step = rsu_env.current_step

    # Check that environment is resetting next headway properly
    assert rsu_env.df.loc[new_time_step + 1, 'Headway'] / max_headway == next_headway / max_headway

    # Check that environment is resetting next velocity properly
    assert rsu_env.df.loc[new_time_step + 1, 'Velocity'] / max_velocity == next_velocity / max_velocity


@pytest.mark.parametrize("action",
                         [(np.array([-1, 0.5, -0.75, 0.3]))])
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
    # Doing this only to get rid of an IDE glitch that does not
    # let me access the dataframe of the RSUEnv()
    if not isinstance(rsu_env, RSUEnv):
        raise Exception("Wrong environment")

    # Apply action array to instantiated RSUEnv
    rsu_env._take_action(action)

    for index, row in rsu_env.df.iterrows():
        # Case where RSU tells the vehicle to speed up
        if action[index] > 0:
            assert (rsu_env.df.at[index, 'Velocity'] - action[index]) == 2
        # Case where the RSU tells the vehicle to slow down
        elif action[index] < 0:
            assert (rsu_env.df.at[index, 'Velocity'] + action[index]) == 2
        # Do nothing case where RSU is satisfied with current speed of vehicle
        else:
            pass


def test_step_func(rsu_env):  #, action):
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
