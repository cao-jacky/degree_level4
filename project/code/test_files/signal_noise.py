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
    
    x_data = np.arange(0, x_length, 1)
    y_data = noise

    signal_to_noise = y_data / np.std(noise)
    print(np.average(signal_to_noise))
        
    plt.figure()
    plt.plot(x_data, y_data, linewidth=0.5, color="#000000")

    plt.xlabel(r'\textbf{x-axis}', fontsize=15)
    plt.ylabel(r'\textbf{y-axis}', fontsize=15)

    plt.tight_layout()
    plt.savefig("signal_noise_graphs/"+str(continuum_level)+".pdf")
    plt.close("all")

    return {'x': x_data, 'y': y_data}

def multiple_spectrums():
    original_galaxy = original_signal_generator()
    original_x = original_galaxy['x']
    original_y = original_galaxy['y']

    runs = 100

    # data to store
    # 1st dimension: number of runs
    # 2nd dimension: columns of data to store
    #   [0]: current index in runs
    #   [1]: storing signal to noise value

    data = np.zeros([runs, 2])

    spectrum = original_y

    for i in range(runs):
        new_noise = np.random.normal(0, 1, len(original_x))
        spectrum = spectrum + new_noise

        new_signal = spectrum
        new_noise = np.std(spectrum)

        new_sn = new_signal / new_noise
        new_sn = np.average(new_sn)

        print(new_sn)

        data[i][0] = i
        data[i][1] = new_sn

        plt.figure()
        plt.plot(original_x, new_signal, linewidth=0.5, color="#000000")

        plt.xlabel(r'\textbf{x-axis}', fontsize=15)
        plt.ylabel(r'\textbf{y-axis}', fontsize=15)

        plt.tight_layout()
        plt.savefig("signal_noise_graphs/"+str(i)+".pdf")
        plt.close("all")

    plt.figure()
    plt.plot(data[:,0], data[:,1], linewidth=0.5, color="#000000")

    plt.xlabel(r'\textbf{x-axis}', fontsize=15)
    plt.ylabel(r'\textbf{y-axis}', fontsize=15)

    plt.tight_layout()
    plt.savefig("signal_noise_results/index_vs_sn.pdf")
    plt.close("all")

#original_signal_generator()
multiple_spectrums()
