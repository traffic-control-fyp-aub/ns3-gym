

import sys
import statistics
import matplotlib.pylab as plt

rsu_data_file = "rsu_0.log"
rewards_file = "rewards.log"

speeds_avg = {}
speeds_std = {}
headways_avg = {}
headways_std = {}
rewards = {}


with open(rsu_data_file,'r') as f1,  open(rewards_file, 'r') as f2:
    line = f1.readline()
    time = 0
    # max_time = 9000
    speeds = []
    headways = []
    try:
        while line:
            if "time" in line:
                time = float(line[line.find("=")+1 : line.find(":")].strip())
                reward = float(f2.readline().split(':')[1].strip())
                rewards[time] = reward

                # if (time>max_time):
                #     break

            line = f1.readline()

            while "time" not in line:
                vals = line.split("::")
                if len(vals) > 1:
                    speeds.append(float(vals[1].strip()))
                    headways.append(float(vals[2].strip()))
                line = f1.readline()

            speeds_avg[time] = statistics.mean(speeds)
            speeds_std[time] = statistics.stdev(speeds)
            headways_avg[time] = statistics.mean(headways)
            headways_std[time] = statistics.stdev(headways)
            speeds = []
            headways = []

    except StopIteration:
        print('EOF!')
    finally:
      
        lists = sorted(speeds_avg.items())
        t, avg = zip(*lists)

        fig = plt.figure(1)
        fig.suptitle('Velocity', fontsize=16)
        plt.plot(t, avg, 'b-', label="Average")

        lists = sorted(speeds_std.items())
        t, std = zip(*lists)
        plt.plot(t, std,'r--', label="Standard Deviation")

        plt.legend()
        # plt.show()

        fig = plt.figure(2)
        fig.suptitle('Headways', fontsize=16)

        lists = sorted(headways_avg.items())
        t, avg = zip(*lists) 
        plt.plot(t, avg, 'b-', label="Average")

        lists = sorted(headways_std.items())
        t, std = zip(*lists)
        plt.plot(t, std,'r--', label="Standard Deviation")
       
        plt.legend()

        fig = plt.figure(3)
        fig.suptitle('Rewards', fontsize=16)

        lists = sorted(rewards.items())
        t, vals = zip(*lists) 
        plt.plot(t, vals, 'g-', label="Reward")
        plt.legend()

        plt.show()


