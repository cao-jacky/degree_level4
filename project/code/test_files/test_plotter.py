import numpy as np

import matplotlib.pyplot as plt

galaxy_data = np.load("cube_5_y.npy")
length = len(galaxy_data)
x_data = np.arange(0, length, 1)

print(galaxy_data)

plt.figure()
plt.plot(x_data, galaxy_data)
plt.savefig("test_cube_5.pdf")
