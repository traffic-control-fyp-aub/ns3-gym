"""
    Usage:
    ------
    To be able to use this file to collect ns3 logs you need two terminal windows and need to run the following
    commands in each window

    Terminal 1 (ns3 side):
    >> ./waf --run "scratch/ns3-sumo-coupling-simple --scenario=[1 | 2 | 3]" 2>&1 | awk '{print > "rl_fyp/ns3_logs/logs.log"}/RSU0 table/{print > "rl_fyp/ns3_logs/rsu_0.log"}/MyGetReward/{print > "rl_fyp/ns3_logs/rewards.log"}'

    Terminal 2 (Gym side):
    >> python3 script.py test scenario=[ scenario_name ] cars=[ number of cars previously trained on ]
"""

import sys
import statistics
import matplotlib.pylab as plt

rewards_file = "rewards.log"
rewards = {}


with open(rewards_file, 'r') as f:
    line = f.readline()
    num = 0
    try:
        while line:
            reward = float(f.readline().split(':')[1].strip())
            rewards[num] = reward
            num = num+1
            print(num)

    except StopIteration:
        print('EOF!')
    finally:
        print(rewards)
        fig = plt.figure(1)
        fig.suptitle('Rewards', fontsize=16)

        lists = sorted(rewards.items())
        t, vals = zip(*lists)
        plt.plot(t, vals, 'g-', label="Reward")
        plt.legend()

        plt.show()
