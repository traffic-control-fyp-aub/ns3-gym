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
import numpy as np


rsu_data_file = ''

argumentList = sys.argv

if argumentList.__len__() < 2:
    print("Please enter RSU log file as argument")
    exit(0)
else:
    rsu_data_file = argumentList[1]

speeds_avg = {}
speeds_std = {}
headways_avg = {}
headways_std = {}


with open(rsu_data_file, 'r') as f:
    line = f.readline()
    time = 0
    speeds = []
    headways = []
    try:
        while line:
            if "time" in line:
                if time > 0:
                    speeds_avg[time] = 0 if len(
                        speeds) < 1 else statistics.mean(speeds)
                    speeds_std[time] = 0 if len(
                        speeds) < 2 else statistics.stdev(speeds)
                    headways_avg[time] = 0 if len(
                        headways) < 1 else statistics.mean(headways)
                    headways_std[time] = 0 if len(
                        headways) < 2 else statistics.stdev(headways)
                    speeds = []
                    headways = []

                time = float(line[line.find("=")+1: line.find(":")].strip())

            elif "time" not in line:
                vals = line.split("::")
                if len(vals) > 1:
                    speeds.append(float(vals[1].strip()))
                    headways.append(float(vals[2].strip()))

            line = f.readline()

    except StopIteration:
        print('EOF!')
    finally:

        print(speeds_avg)
        print(speeds_std)
        print(headways_avg)
        print(headways_std)

        fig = plt.figure(1)
        fig.suptitle('Velocity', fontsize=16)

        lists = sorted(speeds_avg.items())
        t, avg = zip(*lists)
        plt.plot(t, avg, 'b-', label="Average/second")

        t = t[100:]
        avg = avg[100:]
        pfit = np.polyfit(t, avg, 1)
        trend_line_model = np.poly1d(pfit)
        plt.plot(t, trend_line_model(t), "g--",  label="System Average Trend")

        lists = sorted(speeds_std.items())
        t, std = zip(*lists)
        plt.plot(t, std, 'r-', label="Standard Deviation")

        plt.legend()

        fig = plt.figure(2)
        fig.suptitle('Headways', fontsize=16)

        lists = sorted(headways_avg.items())
        t, avg = zip(*lists)
        plt.plot(t, avg, 'b-', label="Average")

        t = t[100:]
        avg = avg[100:]
        pfit = np.polyfit(t, avg, 1)
        trend_line_model = np.poly1d(pfit)
        plt.plot(t, trend_line_model(t), "g--",  label="System Average Trend")

        lists = sorted(headways_std.items())
        t, std = zip(*lists)
        plt.plot(t, std, 'r-', label="Standard Deviation")

        plt.legend()
        plt.show()
