# This file is a utility function file that will help set up the training
# agent based on the user specification in the CLI. It is a helper file to
# the main training/testing script file located at:
#
#                           rl_fyp/gym_rsu/script.py
import inspect
from stable_baselines import *

list_of_algorithms = ['TD3', 'DDPG', 'PPO2', 'PPO1',
                      'A2C', 'ACER', 'ACKTR', 'DQN',
                      'GAIL', 'HER', 'SAC', 'TRPO']


def model_setup(algorithm_name, env, policy, **kwargs):
    """
        This function takes in the algorithm name specified by the
        user in the CLI, the environment to train on, the policy, and
        finally an optional dictionary of parameters to set for the agent.

    Parameter(s):
    -------------
    algorithm_name: type(String)
        The name of the algorithm to be used. Must be supported in stable baselines.
    env: type(Gym)
        The environment to train the agent on. Must be of type openAI Gym.
    policy: type(String)
        The policy to train the agent with. Not all policies are compatible with
        all algorithms.
    kwargs: type(dict)
        Dictionary of algorithm variables that can be used instead of the default values.

    Returns:
    --------
    model: type(Object)
        A machine learning model
    """
    assert algorithm_name in list_of_algorithms, 'Algorithm must be supported by stable baselines'

    model = None

    if algorithm_name in [list_of_algorithms[0]]:
        # TD3 algorithm
        # Get the default values in case no user specifications
        signature = inspect.signature(TD3.__init__)

        g = kwargs.pop('g', signature.parameters['gamma'].default)
        bf = kwargs.pop('bf', signature.parameters['buffer_size'].default)
        nstd = kwargs.pop('nstd', signature.parameters['target_policy_noise'].default)
        lst = kwargs.pop('lst', signature.parameters['learning_starts'].default)
        bch = kwargs.pop('bch', signature.parameters['batch_size'].default)
        lr = kwargs.pop('lr', signature.parameters['learning_rate'].default)
        tf = kwargs.pop('tf', signature.parameters['train_freq'].default)
        grad = kwargs.pop('grad', signature.parameters['gradient_steps'].default)
        pkwargs = kwargs.pop('pkwargs', signature.parameters['policy_kwargs'].default)
        v = kwargs.pop('v', signature.parameters['verbose'].default)

        model = TD3(policy=policy,
                    env=env,
                    gamma=g,
                    buffer_size=bf,
                    target_policy_noise=nstd,
                    learning_starts=lst,
                    batch_size=bch,
                    learning_rate=lr,
                    train_freq=tf,
                    gradient_steps=grad,
                    verbose=v,
                    policy_kwargs=pkwargs)

    elif algorithm_name in [list_of_algorithms[1]]:
        # DDPG algorithm
        pass
    elif algorithm_name in [list_of_algorithms[2]]:
        # PPO2 algorithm
        # Get the default values in case no user specifications
        signature = inspect.signature(PPO2.__init__)

        lr = kwargs.pop('lr', signature.parameters['learning_rate'].default)
        v = kwargs.pop('v', signature.parameters['verbose'].default)
        ent = kwargs.pop('ent', signature.parameters['ent_coef'].default)
        lbd = kwargs.pop('lbd', signature.parameters['lam'].default)
        g = kwargs.pop('g', signature.parameters['gamma'].default)

        model = PPO2(policy=policy,
                     env=env,
                     learning_rate=lr,
                     verbose=v,
                     ent_coef=ent,
                     lam=lbd,
                     gamma=g)

    elif algorithm_name in [list_of_algorithms[3]]:
        # PPO1 algorithm
        pass
    elif algorithm_name in [list_of_algorithms[4]]:
        # A2C algorithm
        pass
    elif algorithm_name in [list_of_algorithms[5]]:
        # ACER algorithm
        pass
    elif algorithm_name in [list_of_algorithms[6]]:
        # ACKTR algorithm
        pass
    elif algorithm_name in [list_of_algorithms[7]]:
        # DQN algorithm
        pass
    elif algorithm_name in [list_of_algorithms[8]]:
        # GAIL algorithm
        pass
    elif algorithm_name in [list_of_algorithms[9]]:
        # HER algorithm
        pass
    elif algorithm_name in [list_of_algorithms[10]]:
        # SAC algorithm
        # Get the default values in case no user specifications
        signature = inspect.signature(SAC.__init__)

        lr = kwargs.pop('lr', signature.parameters['learning_rate'].default)
        bf = kwargs.pop('bf', signature.parameters['buffer_size'].default)
        bch = kwargs.pop('bch', signature.parameters['batch_size'].default)
        ent = kwargs.pop('ent', signature.parameters['ent_coef'].default)
        tf = kwargs.pop('tf', signature.parameters['train_freq'].default)
        grad = kwargs.pop('grad', signature.parameters['gradient_steps'].default)
        lst = kwargs.pop('lst', signature.parameters['learning_starts'].default)
        v = kwargs.pop('v', signature.parameters['verbose'].default)

        model = SAC(policy=policy,
                    env=env,
                    learning_rate=lr,
                    buffer_size=bf,
                    batch_size=bch,
                    ent_coef=ent,
                    train_freq=tf,
                    gradient_steps=grad,
                    learning_starts=lst,
                    verbose=v)

    elif algorithm_name in [list_of_algorithms[11]]:
        # TRPO algorithm
        pass

    return model
