import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np

from lmfit import Parameters, Model

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def gauss(x, a, scale, mean, sigma):
    sigma_term = 1 / (sigma*np.sqrt(2*np.pi))
    exp_term = np.exp(-(1/2)*((x-mean)/(sigma))**2)
    return a + scale*sigma_term*exp_term

def original_signal_generator():
    np.random.seed(0) # seeding the random numpy generator to create an original signal

    x_length = 1000
    x_data = np.arange(0, x_length, 1)
    
    continuum_level = 100

    # adding a gaussian profile to the data
    gauss_std = 10
    scale = 10.0
    mean = x_length/2
    y_gauss = gauss(x_data, continuum_level, scale, mean, gauss_std)
 
    noise = np.random.normal(0, 0.25, x_length)
    #y_data = np.full(x_length, continuum_level)
    y_data = y_gauss + noise

    original_params = {'a': continuum_level, 'scale': scale, 'mean': mean,
            'sigma': gauss_std}

    # refitting using lmfit to verify parameter values
    gauss_params = Parameters()
    gauss_params.add('a', value=continuum_level)
    gauss_params.add('scale', value=scale, min=0.0)
    gauss_params.add('mean', value=mean)
    gauss_params.add('sigma', value=gauss_std, vary=False)

    gauss_model = Model(gauss)
    gauss_result = gauss_model.fit(y_data, x=x_data, params=gauss_params)

    optimal_params = gauss_result.best_values
    print(optimal_params)

    model_gauss = gauss(x_data, optimal_params['a'], optimal_params['scale'], 
            optimal_params['mean'], optimal_params['sigma']) 
    
    signal_to_noise = y_data / np.std(y_data)
    print(np.average(signal_to_noise))
 
    plt.figure()
    plt.plot(x_data, y_data, linewidth=0.5, color="#000000")
    plt.plot(x_data, model_gauss, linewidth=0.5, color="#e53935")
    plt.plot(x_data, y_gauss, linewidth=0.5, color="#ffc107")

    plt.xlabel(r'\textbf{x-axis}', fontsize=15)
    plt.ylabel(r'\textbf{y-axis}', fontsize=15)

    plt.tight_layout()
    plt.savefig("signal_noise_graphs/signal_original.pdf")
    plt.close("all")

    return {'x': x_data, 'y': y_data, 'gauss_parameters': optimal_params, 
            'original_parameters': original_params}

def multiple_spectrums():
    original_galaxy = original_signal_generator()
    original_x = original_galaxy['x']
    original_y = original_galaxy['y']

    best_gauss_params = original_galaxy['gauss_parameters']
    bgm = best_gauss_params

    best_sigma = bgm['sigma']

    runs = 200

    # data to store
    # 1st dimension: number of runs
    # 2nd dimension: columns of data to store
    #   [0]: current index in runs
    #   [1]: storing signal to noise value
    #   [2]: storing new sigma of gaussian
    #   [3]: storing abs(sigma_in-sigma_out)/sigma_in

    data = np.zeros([runs, 4])

    spectrum = original_y

    for i in range(runs):
        np.random.seed()
        new_noise = 20*np.random.normal(0, 0.25, len(original_x))
        #print(np.std(spectrum))
        spectrum = spectrum + new_noise

        # fitting a gaussian to the new spectrum
        new_gauss_params = Parameters()
        new_gauss_params.add('a', value=bgm['a'])
        new_gauss_params.add('scale', value=bgm['scale'], min=0.0)
        new_gauss_params.add('mean', value=bgm['mean'], min=bgm['mean']-5, 
                max=bgm['mean']+5)
        new_gauss_params.add('sigma', value=bgm['sigma'], min=9.9)

        new_gauss_model = Model(gauss)
        new_gauss_result = new_gauss_model.fit(spectrum, x=original_x, 
                params=new_gauss_params)

        new_optimal_params = new_gauss_result.best_values
        print(new_optimal_params)

        new_sigma = new_optimal_params['sigma']
        sigma_diff = np.absolute(best_sigma-new_sigma)/best_sigma

        data[i][2] = new_sigma
        data[i][3] = sigma_diff
        print(sigma_diff)

        new_model_gauss = gauss(original_x, new_optimal_params['a'], 
                new_optimal_params['scale'], new_optimal_params['mean'], 
                new_optimal_params['sigma'])

        new_signal = spectrum
        new_noise_std = np.std(spectrum)
        #print(np.std(new_noise), new_noise_std)

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
        plt.plot(original_x, new_model_gauss, linewidth=0.5, color="#b2ff59")

        plt.xlabel(r'\textbf{x-axis}', fontsize=15)
        plt.ylabel(r'\textbf{signal}', fontsize=15)

        plt.tight_layout()
        plt.savefig("signal_noise_graphs/signal_"+str(i)+".pdf")
        plt.close("all")

    # plotting index vs S/N
    plt.figure()
    plt.plot(data[:,1], data[:,0], linewidth=0.5, color="#000000")

    plt.xlabel(r'\textbf{signal to noise}', fontsize=15)
    plt.ylabel(r'\textbf{index}', fontsize=15)

    plt.tight_layout()
    plt.savefig("signal_noise_results/index_vs_sn.pdf")
    plt.close("all")

    # plotting abs(sigma_in-sigma_out)/sigma_in vs S/N
    plt.figure()
    plt.plot(data[:,1], data[:,3], linewidth=0.5, color="#000000")

    plt.xlabel(r'\textbf{signal to noise}', fontsize=15)
    plt.ylabel(r'\textbf{$\mid\sigma_{in}-\sigma_{out}\mid$/$\sigma_{in}$}', fontsize=15)

    plt.tight_layout()
    plt.savefig("signal_noise_results/sigma_diff_vs_sn.pdf")
    plt.close("all")

    # plotting abs(sigma_in-sigma_out)/sigma_in vs S/N from 10 to 0 
    plt.figure()
    plt.plot(data[:,1], data[:,3], linewidth=0.5, color="#000000")

    plt.xlabel(r'\textbf{signal to noise}', fontsize=15)
    plt.ylabel(r'\textbf{$\mid\sigma_{in}-\sigma_{out}\mid$/$\sigma_{in}$}', fontsize=15)

    plt.xlim([0,10])
    plt.tight_layout()
    plt.savefig("signal_noise_results/sigma_diff_vs_sn_0to10.pdf")
    plt.close("all")


#original_signal_generator()
multiple_spectrums()
