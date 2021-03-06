
"""
    This is the main set up file for the custom gym environment.
    Whatever is entered as a name here is what will be used
    to import the custom environment later at the top of a file.
    Usage:
    ------
    import gym_rsu
"""
from setuptools import setup

setup(name='gym_rsu',
      version='0.0.1',
      install_requires=['gym',
                        'baselines']
      )
