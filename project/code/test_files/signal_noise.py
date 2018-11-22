import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def original_signal_generator():
    np.random.seed(0) # seeding the random numpy generator to create an original signal

    x_length = 1000
    x_data = np.arange(0, x_length, 1)

    continuum_level = 100
    noise = np.random.normal(0, 1, x_length)

    y_data = np.full(x_length, continuum_level)
    y_data = y_data + noise

    signal_to_noise = y_data / np.std(y_data)
    print(np.average(signal_to_noise))
 
    plt.figure()
    plt.plot(x_data, y_data, linewidth=0.5, color="#000000")

    plt.xlabel(r'\textbf{x-axis}', fontsize=15)
    plt.ylabel(r'\textbf{y-axis}', fontsize=15)

    plt.tight_layout()
    plt.savefig("signal_noise_graphs/signal_"+str(continuum_level)+".pdf")
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
        np.random.seed()
        new_noise = np.random.normal(0, 1, len(original_x))
        print(np.std(spectrum))
        spectrum = spectrum + new_noise

        new_signal = spectrum
        new_noise_std = np.std(spectrum)
        print(np.std(new_noise), new_noise_std)

        new_sn = new_signal / new_noise_std
        new_sn = np.average(new_sn)

        print(new_sn)

        data[i][0] = i
        data[i][1] = new_sn

        plt.figure()
        plt.plot(original_x, new_noise, linewidth=0.5, color="#000000")

        plt.xlabel(r'\textbf{x-axis}', fontsize=15)
        plt.ylabel(r'\textbf{noise}', fontsize=15)

        plt.tight_layout()
        plt.savefig("signal_noise_noise/noise_"+str(i)+".pdf")
        plt.close("all")

        plt.figure()
        plt.plot(original_x, new_signal, linewidth=0.5, color="#b71c1c")
        plt.plot(original_x, original_y, linewidth=0.5, color="#000000")


        plt.xlabel(r'\textbf{x-axis}', fontsize=15)
        plt.ylabel(r'\textbf{signal}', fontsize=15)

        plt.tight_layout()
        plt.savefig("signal_noise_graphs/signal_"+str(i)+".pdf")
        plt.close("all")

    plt.figure()
    plt.plot(data[:,1], data[:,0], linewidth=0.5, color="#000000")

    plt.xlabel(r'\textbf{signal to noise}', fontsize=15)
    plt.ylabel(r'\textbf{index}', fontsize=15)

    plt.tight_layout()
    plt.savefig("signal_noise_results/index_vs_sn.pdf")
    plt.close("all")

#original_signal_generator()
multiple_spectrums()
