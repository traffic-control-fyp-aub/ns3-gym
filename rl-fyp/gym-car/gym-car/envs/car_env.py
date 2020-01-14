import gym
from gym import error, spaces, utils
from gym.utils import seeding


"""
    Custom OpenAI Gym environment from the perspective
    of the car.
"""
class CarEnv(gym.Env):
    metadata = {'render.modes':['human']}

    
    def __init__(self):
        """
            Initialization costructor for the
            custom environment.
        """
        pass

    def step(self, action):
        """
            Step function to be taken on the environment.

            Parameter(s):
            -------------
            action: Object
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
        pass

    def reset(self):
        """
            Reset function that sets the environment back
            to it's initial settings.
        """
        pass

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
        pass

