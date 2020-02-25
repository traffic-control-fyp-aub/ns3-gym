import numpy as np
import pandas as pd
import math

"""
    This is a helper file to assist in
    generating data to train the RSUEnv
"""

MEAN_VELOCITY = 92  # value to center normal distribution velocity sampling
MEAN_HEADWAY = 1.5  # value to center normal distribution headway sampling
SIGMA = 0.1  # standard deviation for normal distribution velocity/headway sampling
PATH_TO_CSV = "rl_fyp/training_data/training_data_frame.csv"  # path to where the dataframe was saved as a CSV file
DIRECT_PATH_TO_CSV = "/home/rayyan/Desktop/FYP/repos/ns3-gym/rl_fyp/training_data/training_data_frame.csv"

# Start off by drawing samples from a normal
# distribution centered at MEAN_VELOCITY with
# standard deviation SIGMA. These values are
# saved to be used later as initial velocities
# while training an agent on the RSUEnv.

velocities = np.asarray([])
for _ in range(10):
    # Generate 10,000 samples of velocities from a normal distribution
    # centered around the mean velocity with standard deviation sigma.
    velocities = np.append(velocities, round(abs(np.random.normal(MEAN_VELOCITY, SIGMA)), 2))

headways = np.asarray([])
for _ in range(10):
    # Generate 10,000 samples of headways from a normal distribution
    # centered around the mean headway with standard deviation sigma.
    headways = np.append(headways, round(abs(np.random.normal(MEAN_HEADWAY, SIGMA)), 2))

# Two dimentional dataframe consisting of the sampled headways and velocities
training_dataframe = pd.DataFrame({'Headway': headways, 'Velocity': velocities})

# Export the dataframe containing all the training data
# as a CSV file located at PATH_TO_CSV
try:
    print("Trying the relative path")
    export_csv = training_dataframe.to_csv(PATH_TO_CSV)
    print("** Success **")
except FileNotFoundError:
    print("Trying the direct path")
    export_csv = training_dataframe.to_csv(DIRECT_PATH_TO_CSV)
    print("** Success **")
