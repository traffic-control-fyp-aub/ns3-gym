"""
    The id variable entered here is what we use when
    we want to call the gym.make() function.

    Usage:
    ------
    gym.make('car-v0')
"""

from gym.envs.registration import register

register(
        id='car_v0',
        entry_point='gym_car.envs:CarEnv',
        )
