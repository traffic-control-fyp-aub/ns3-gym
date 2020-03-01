

import sys
import statistics
import matplotlib.pylab as plt

infile = "rsu_0.log"
speeds_avg = {}
speeds_std = {}
headways_avg = {}
headways_std = {}

with open(infile) as fp:
    line = fp.readline()
    time = 0
    max_time = 9000
    speeds = []
    headways = []
    try:
        while line:
            if "time" in line:

                time = float(line[line.find("=")+1 : line.find(":")].strip())
                
                if (time>max_time):
                    break

            line = fp.readline()

            while "time" not in line:
                vals = line.split("::")
                if len(vals) > 1:
                    speeds.append(float(vals[1].strip()))
                    headways.append(float(vals[2].strip()))
                line = fp.readline()

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
        plt.plot(t, avg, 'g', label="Average")

        lists = sorted(speeds_std.items())
        t, std = zip(*lists)
        plt.plot(t, std,'r', label="Standard Deviation")

        # fig = plt.figure()
        # fig.suptitle('Velocity', fontsize=16)
        plt.legend()
        plt.show()

        lists = sorted(headways_avg.items())
        t, avg = zip(*lists) 
        plt.plot(t, avg, 'g', label="Average")

        lists = sorted(headways_std.items())
        t, std = zip(*lists)
        plt.plot(t, std,'r', label="Standard Deviation")
        # fig = plt.figure()
        # fig.suptitle('Headways', fontsize=16)
        plt.legend()
        plt.show()


