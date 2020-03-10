import random
import matplotlib.pyplot as plt
import numpy as np

MAX_VELOCITY_VALUE = 100
DESIRED_VELOCITY_VALUE = 95
ALPHA = 0.1

dict_of_speeds = {}
temp = []
low = 30
high = 40
for i in range(7):
	for j in range(4):
		temp.append(random.randint(low, high))
	dict_of_speeds[i] = temp
	temp = []
	low += 10
	high += 10

list_of_maximums = []
for z in range(4):
	list_of_maximums.append(random.randint(1,2))

x_vals = []
y_vals = []

for key, value in dict_of_speeds.items():
	avg = sum(value)/len(value)  # get the avg value on interval
	x_vals.append(avg)
	
	reward = abs(MAX_VELOCITY_VALUE) - abs((np.sum(np.subtract(np.array([DESIRED_VELOCITY_VALUE, DESIRED_VELOCITY_VALUE, DESIRED_VELOCITY_VALUE, DESIRED_VELOCITY_VALUE]), np.asarray(value)))))/4 - ALPHA*(sum(np.asarray(list_of_maximums)))
	
	y_vals.append(reward)

plt.plot(x_vals, y_vals)

plt.xlabel('speeds')
plt.ylabel('rewards')
plt.title('Reward values')

plt.show()

