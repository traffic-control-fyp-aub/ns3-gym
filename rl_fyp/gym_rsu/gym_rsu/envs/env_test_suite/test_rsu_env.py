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


@pytest.mark.parametrize("obs_space, action_space, reward", [(gym.spaces.Box(0,
                                                                             100,
                                                                             shape=(20,),
                                                                             dtype=np.float16),
                                                              gym.spaces.Box(-10,
                                                                             10,
                                                                             shape=(10,),
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


@pytest.mark.parametrize("next_headway, next_velocity, max_headway, max_velocity, epsilon",
                         [(1, 90, 2, 100, 35)])
def test_next_observation(rsu_env, next_headway, next_velocity, max_headway, max_velocity, epsilon):
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

    obs_array = rsu_env._next_observation()

    # Next time step headway value
    for index in range(int(len(obs_array)/2)):
        assert math.pow(abs(obs_array[index] - next_headway), 2) <= epsilon

    # Next time step velocity value
    for index in range(int(len(obs_array) / 2), int(len(obs_array))):
        assert math.pow(abs(obs_array[index] - next_velocity), 2) <= epsilon


@pytest.mark.parametrize("next_headway, next_velocity, max_headway, max_velocity, epsilon",
                         [(1, 90, 2, 100, 35)])
def test_env_reset(rsu_env, next_headway, next_velocity, max_headway, max_velocity, epsilon):
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
    obs = rsu_env.reset()

    # Check that environment is resetting next headway properly
    for index in range(10):
        assert math.pow(abs(obs[index] - round(abs(np.random.normal(1.5, 0.1)), 2)), 2) <= epsilon

    # Check that environment is resetting next velocity properly
    for index in range(10, 20):
        assert math.pow(abs(obs[index] - round(abs(np.random.normal(90, 0.1)), 2)), 2) <= epsilon


@pytest.mark.parametrize("action, epsilon",
                         [(np.array([-10, 5, -7.5, 3,
                                     -10, 5, -7.5, 3,
                                     -10, 5, -7.5, 3,
                                     -10, 5, -7.5, 3,
                                     -10, 5, -7.5, 3]), 35)])
def test_take_action(rsu_env, action, epsilon):
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

    for index, _ in rsu_env.df.iterrows():
        assert math.pow(abs(rsu_env.df.at[index, 'Velocity'] - (100 + action[index])), 2) <= epsilon


@pytest.mark.parametrize("obs_vel, reward, done, epsilon",
                         [(np.array([90, 90, 90, 90,
                                     90, 90, 90, 90,
                                     90, 90, 90, 90,
                                     90, 90, 90, 90,
                                     90, 90, 90, 90]),
                          35,
                          False,
                           35)])
def test_step_func(rsu_env, obs_vel, reward, done, epsilon):
    """
        Test the step function in the RSUEnv.
        Parameter(s):
        -------------
        rsu_env: type(gym.Env)
            Instantiated object of the custom RSUEnv class.
        action: type(Numpy Array)
            Array of de/acceleration values for the vehicles
            present in the RSUEnv
        obs_h: type(Numpy Array)
            Array of headway times for the next time
            step of length N where:
                - N = number of vehicles on circuit
        obs_vel: type(Numpy Array)
            Array of velocities for the next time
            step of length N where:
                - N = number of vehicles on circuit
        reward: type(Float)
            Reward collected from the environment
            based on the current step taken.
        done: type(Bool)
            Boolean value that tells the agent
            whether the environment specific
            end condition has been reached.
    """
    # Doing this only to get rid of an IDE glitch that does not
    # let me access the dataframe of the RSUEnv()
    if not isinstance(rsu_env, RSUEnv):
        raise Exception("Wrong environment")

    obs, rew, d, _ = rsu_env.step()

    # Assert that the done condition is correct
    assert d == done

    # Assert that the reward is close to what is expected
    # because we are re-sampling for the headway times
    # so there is some uncertainty and we only specify
    # and upper limit on the expected reward.
    if rew == -10:
        pass
    else:
        assert math.pow(abs(reward - rew), 2) <= epsilon

    # Returned observation is of length 2N where N is
    # the number of vehicles on the circuit. The first
    # half of the returned observations is the observed
    # headway times and the second half is the observed
    # velocities.
    velocities = np.asarray([])

    # Splitting the returned observations into the
    # headways and velocities
    for index in range(int(len(obs)/2), len(obs)):
        velocities = np.append(velocities, obs[index])

    # Assert the velocities
    for index in range(len(velocities)):
        assert math.pow(abs(velocities[index] - obs_vel[index]), 2) <= epsilon
