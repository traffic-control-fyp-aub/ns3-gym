# Custom OpenAI Gym Environment

This environment is a custom gym environment for the sole purpose of training the agent.

## Environment Installation

To install and register the environment with OpenAI Gym run the following script in the working directory of the **setup.py** file.

```bash 
pip install -e .
```

## Environment Usage

After installing and registering the custom environment you can use it with the following code block:

```bash 
import gym
import rl-fyp.gym-rsu

env = gym.make('rsu-v0')
```
