

import numpy as np
import matplotlib.pyplot as plt

size=100
randomSet = np.zeros(size)
xAxis = np.linspace(0,100,size)


for i in range(0,size):
    randomPoint = np.random.rand(100)
    mean = np.sum(randomPoint)/len(randomPoint)
    randomSet[i] = mean

plt.hist(randomSet)
plt.show()