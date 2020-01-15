"""
    The id variable entered here is what we use when
    we want to call the gym.make() function.

    Usage:
    ------
    gym.make('rsu-v0')
"""

from gym.envs.registration import register

register(
        id='rsu-v0',
        entry_point='gym_rsu.envs:RSUEnv',
        )
