#!/usr/bin/env python3

"""
    This file is a custom shell script written in Python to handle the entire training
    process by directing the user through the various adaptable learning algorithms and
    and features of NGS.

    Usage:
    ------
    Within the current directory of the `bash.py` file enter the following command into the terminal:
    >> python3 bash.py

    Once entered then the user is working within the context of the custom NGS shell and no longer their own.
    To exit from the shell either type in the word `exit` into the custom shell of click on CTRL + C.

    Command(s):
    ----------
    - exit: Terminate the custom NGS shell and return the user to their own shell.

    - help: List out all the commands supported by the custom NGS shell and their functions
            (including the `help` command).

    - train: Start the training process of a machine learning algorithm. User is later asked
             to specify if they would like to train `online` via connecting directly to ns3-
             and SUMO or to train `offline` by connecting to the custom OpenAI Gym environment
             called `RSUEnv` and not utilize the bridge to ns3-SUMO.

    - test: Start the testing process of a previously trained machine learning algorithm.

    Command Option(s):
    ------------------
    List out the name and description of the option as well as what command said option belongs to.

    - online: option for `train` command
        This option tells NGS that the user wishes to connect to ns3-SUMO via the ZMQ bridge
        and perform training using live vehicle data sent over the bridge.

    - offline: option for `train` command
        This option tells NGS that the user wishes to only use OpenAI Gym and train on
        previously collected sample of data.

    - scenario: option for `train` and `test` commands
        This option takes the name of the scenario to be loaded into ns3-SUMO.

    - algorithm: option for 'train` w/ `online` option previously selected
        This option takes in the name of the algorithm that the user wishes to perform
        training on. Currently only supported for online training option. List of RL
        algorithms supported are all those within the stable-baselines repository:
            + A2C
            + ACER
            + ACKTR
            + DDPG
            + DQN
            + GAIL
            + HER
            + PPO1
            + PPO2
            + SAC
            + TD3
            + TRPO

    - policy_kwargs: option for `train` w/ `online` option selected
        This option is *optional* and depends on whether the user wishes to specify their own
        set of parameter values for the selected policy algorithm that is being used for training.
        Some example parameters could be `learning_rate` and `batch_size`. The user may pass on specifying
        any options simply by typing in the keyword `pass`.
"""

import os
import subprocess

import training_util_script


def execute_cmd(command):
    """
        Helper function for the execution of shell commands

    Parameter(s):
    -------------
    command: type(String)
        User specified command in the shell
    """
    pass


def bash_help():
    """
        List out all the commands supported by the custom NGS shell and their functions
        (including the `help` command).
    """
    pass


def main():
    """
        Main point of execution of the shell command. Enters and infinite loop
        and keeps running until either the exit command is specified or CTRl + C.
    """
    pass


if '__main__' == __name__:
    main()
