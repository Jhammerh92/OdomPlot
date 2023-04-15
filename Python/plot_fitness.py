import numpy as np
import matplotlib.pyplot as plt

fitness = np.loadtxt('/home/slamnuc/Desktop/fitness_all_keyframes.csv', delimiter=",", dtype=float, usecols=[0])
print(fitness)

plt.plot(fitness)
plt.show()