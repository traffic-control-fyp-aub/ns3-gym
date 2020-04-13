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
fuel_consumption_avg = {}
co2_emission_avg = {}
co_emission_avg = {}
nox_emission_avg = {}
pmx_emission_avg = {}
hc_emission_avg = {}


with open(rsu_data_file, 'r') as f:
    line = f.readline()
    time = 0
    speeds = []
    headways = []
    fuel_consumption = []
    co2_emission = []
    co_emission = []
    nox_emission = []
    pmx_emission = []
    hc_emission = []

    try:
        while line:
            if "time" in line:
                if time > 0:
                    # save speeds mean and std
                    speeds_avg[time] = 0 if len(
                        speeds) < 1 else statistics.mean(speeds)
                    speeds_std[time] = 0 if len(
                        speeds) < 2 else statistics.stdev(speeds)
                    speeds = []

                    # save headways mean and std
                    headways_avg[time] = 0 if len(
                        headways) < 1 else statistics.mean(headways)
                    headways_std[time] = 0 if len(
                        headways) < 2 else statistics.stdev(headways)
                    headways = []

                    # save fuel consumption mean
                    fuel_consumption_avg[time] = 0 if len(
                        fuel_consumption) < 1 else statistics.mean(fuel_consumption)
                    fuel_consumption = []

                    # save CO2 emission mean
                    co2_emission_avg[time] = 0 if len(
                        co2_emission) < 1 else statistics.mean(co2_emission)
                    co2_emission = []

                    # save CO emission mean
                    co_emission_avg[time] = 0 if len(
                        co_emission) < 1 else statistics.mean(co_emission)
                    co_emission = []

                    # save NOx emission mean
                    nox_emission_avg[time] = 0 if len(
                        nox_emission) < 1 else statistics.mean(nox_emission)
                    nox_emission = []

                    # save PMx mean
                    pmx_emission_avg[time] = 0 if len(
                        nox_emission) < 1 else statistics.mean(pmx_emission)
                    pmx_emission = []

                    # save hc mean
                    hc_emission_avg[time] = 0 if len(
                        hc_emission) < 1 else statistics.mean(hc_emission)
                    hc_emission = []

                time = float(line[line.find("=")+1: line.find(":")].strip())

            elif "time" not in line:
                vals = line.split("::")
                if len(vals) > 1:
                    speeds.append(float(vals[1].strip()))
                    headways.append(float(vals[2].strip()))
                    fuel_consumption.append(float(vals[3].strip()))
                    co2_emission.append(float(vals[4].strip()))
                    co_emission.append(float(vals[5].strip()))
                    nox_emission.append(float(vals[6].strip()))
                    pmx_emission.append(float(vals[7].strip()))
                    hc_emission.append(float(vals[8].strip()))

            line = f.readline()

    except StopIteration:
        print('EOF!')
    finally:

        print(speeds_avg)
        print(speeds_std)
        print(headways_avg)
        print(headways_std)

        # plot speed average, std, trend
        fig = plt.figure(1)
        fig.suptitle('Velocity (m/s)', fontsize=16)

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

        # plot headways average, std, trend
        fig = plt.figure(2)
        fig.suptitle('Headways (s)', fontsize=16)

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

        # plot fuel consumption average
        fig = plt.figure(3)
        fig.suptitle('Fuel Consumption (ml/s)', fontsize=16)

        lists = sorted(fuel_consumption_avg.items())
        t, vals = zip(*lists)
        plt.plot(t, vals, 'r-', label="Fuel Consumption")
        plt.legend()


        # plot co2 emission average
        fig = plt.figure(4)
        fig.suptitle('CO2 Emissions (mg/s)', fontsize=16)
        lists = sorted(co2_emission_avg.items())
        t, vals = zip(*lists)
        plt.plot(t, vals, 'b-', label="Carbon Dioxide")
        plt.legend()

        # plot co emission average
        fig = plt.figure(5)
        fig.suptitle('Carbon Monoxide (mg/s)', fontsize=16)
        lists = sorted(co_emission_avg.items())
        t, vals = zip(*lists)
        plt.plot(t, vals, 'g-', label="Carbon Monoxide")
        plt.legend()

        # plot nox emission average
        fig = plt.figure(6)
        fig.suptitle('NOx Emissions (mg/s)', fontsize=16)
        lists = sorted(nox_emission_avg.items())
        t, vals = zip(*lists)
        plt.plot(t, vals, 'm-', label="Nitrogen Oxides")
        plt.legend()

        # plot pmx emission average
        fig = plt.figure(7)
        fig.suptitle('Light-Duty Emissions (mg/s)', fontsize=16)
        lists = sorted(pmx_emission_avg.items())
        t, vals = zip(*lists)
        plt.plot(t, vals, 'y-', label="Particular Matter")

         # plot hc emission average
        lists = sorted(pmx_emission_avg.items())
        t, vals = zip(*lists)
        plt.plot(t, vals, 'c-', label="HydroCarbons")
                
        plt.legend()
        plt.show()
