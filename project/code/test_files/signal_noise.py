import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def original_signal_generator():
    np.random.seed(0) # seeding the random numpy generator to create an original signal

    x_length = 1000

    continuum_level = 100
    noise = np.random.normal(continuum_level, 1, x_length)
    print(np.std(noise))

    x_data = np.arange(0, x_length, 1)
    y_data = noise
    
    plt.figure()
    plt.plot(x_data, y_data, linewidth=0.5, color="#000000")

    plt.xlabel(r'\textbf{x-axis}', fontsize=15)
    plt.ylabel(r'\textbf{y-axis}', fontsize=15)

    plt.tight_layout()
    plt.savefig("signal_noise_graphs/"+str(continuum_level)+".pdf")
    plt.close("all")


original_signal_generator()

