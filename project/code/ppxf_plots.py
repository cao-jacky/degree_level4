import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
import peakutils

import ppxf_fitter_kinematics_sdss
import cube_analysis

def model_data_overlay(cube_id):
    
    x_model_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_x.npy")  
    y_model = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_model.npy")
    print(x_model_data)
    print(y_model)

    x_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbd_x.npy") 
    y_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbs_y.npy")

    #print(np.shape(x_data), np.shape(y_data), np.shape(y_model))
    
    plt.figure()
    plt.plot(x_model_data, y_model, linewidth=0.5, color="#000000")
    #plt.plot(x_data, y_data, linewidth=0.5, color="#42a5f5")
    plt.savefig("ppxf_results/cube_" + str(int(cube_id)) + "data_model.pdf")

def f_doublet(x, c, i1, i2, sigma_gal, z, sigma_inst):
    """ function for Gaussian doublet """  
    dblt_mu = [3727.092, 3729.875] # the actual non-redshifted wavelengths
    l1 = dblt_mu[0] * (1+z)
    l2 = dblt_mu[1] * (1+z)

    sigma = np.sqrt(sigma_gal**2 + sigma_inst**2)

    norm = (sigma*np.sqrt(2*np.pi))
    term1 = ( i1 / norm ) * np.exp(-(x-l1)**2/(2*sigma**2))
    term2 = ( i2 / norm ) * np.exp(-(x-l2)**2/(2*sigma**2)) 
    return (c*x + term1 + term2)

def fitting_plotter(cube_id):
    # defining wavelength as the x-axis
    x_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_lamgal.npy")

    # defining the flux from the data and model
    y_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_flux.npy")
    y_model = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_model.npy")

    # scaled down y data 
    y_data_scaled = y_data/np.median(y_data)

    # opening cube to obtain the segmentation data
    cube_file = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/cube_"
        + str(cube_id) + ".fits")
    hdu = fits.open(cube_file)
    segmentation_data = hdu[2].data
    seg_loc_rows, seg_loc_cols = np.where(segmentation_data == cube_id)
    signal_pixels = len(seg_loc_rows) 

    # noise spectra will be used as in the chi-squared calculation
    noise = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_noise.npy")
    noise_median = np.median(noise)
    noise_stddev = np.std(noise) 

    residual = y_data_scaled - y_model
    res_median = np.median(residual)
    res_stddev = np.std(residual)

    noise = noise
    
    mask = ((residual < res_stddev) & (residual > -res_stddev)) 
 
    chi_sq = (y_data_scaled[mask] - y_model[mask])**2 / noise[mask]**2
    total_chi_sq = np.sum(chi_sq)

    total_points = len(chi_sq)
    reduced_chi_sq = total_chi_sq / total_points

    print("Cube " + str(cube_id) + " has a reduced chi-squared of " + 
            str(reduced_chi_sq))

    # spectral lines
    sl = {
            'emis': {
                '':             '3727.092', 
                'OII':          '3728.875',
                'HeI':          '3889.0',
                'SII':          '4072.3',
                'H$\delta$':    '4101.89',
                'H$\gamma$':    '4341.68'
                },
            'abs': {
                r'H$\theta$':   '3798.976',
                'H$\eta$':      '3836.47',
                'CaK':          '3934.777',
                'CaH':          '3969.588',
                'G':            '4305.61' 
                },
            'iron': {
                'FeI1':     '4132.0581',
                'FeI2':     '4143.8682',
                'FeI3':     '4202.0293', 
                'FeI4':     '4216.1836',
                'FeI5':     '4250.7871',
                'FeI6':     '4260.4746',
                'FeI7':     '4271.7607',
                'FeI8':     '4282.4028',
                }
            }

    # parameters from lmfit
    cube_result_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) 
            + "_lmfit.txt")
    cube_result_file = open(cube_result_file)

    line_count = 0 
    for crf_line in cube_result_file:
        if (line_count == 15):
            curr_line = crf_line.split()
            c = float(curr_line[1])
        if (line_count == 16):
            curr_line = crf_line.split()
            i1 = float(curr_line[1])
        if (line_count == 18):
            curr_line = crf_line.split()
            i2 = float(curr_line[1])
        if (line_count == 19):
            curr_line = crf_line.split()
            sigma_gal = float(curr_line[1])
        if (line_count == 20):
            curr_line = crf_line.split()
            z = float(curr_line[1])
        if (line_count == 21):
            curr_line = crf_line.split()
            sigma_inst = float(curr_line[1])
        line_count += 1

    plt.figure()

    plt.plot(x_data, y_data_scaled, linewidth=0.1, color="#000000")
    plt.plot(x_data, y_data_scaled+noise_stddev, linewidth=0.1, color="#616161", alpha=0.1)
    plt.plot(x_data, y_data_scaled-noise_stddev, linewidth=0.1, color="#616161", alpha=0.1)
    
    # plotting over the OII doublet
    doublets = np.array([3727.092, 3728.875])
    dblt_av = np.average(doublets) * (1+z)

    dblt_x_mask = ((x_data > dblt_av-20) & (x_data < dblt_av+20))
    doublet_x_data = x_data[dblt_x_mask]
    doublet_data = f_doublet(doublet_x_data, c, i1, i2, sigma_gal, z, sigma_inst)
    doublet_data = doublet_data / np.median(y_data)
    plt.plot(doublet_x_data, doublet_data, linewidth=0.5, color="#9c27b0")

    max_y = np.max(y_data_scaled)
    # plotting spectral lines
    for e_key, e_val in sl['emis'].items():
        spec_line = float(e_val) * (1+z)
        spec_label = e_key

        if (e_val in str(doublets)):
            alpha_line = 0.2
        else:
            alpha_line = 0.7
            
        alpha_text = 0.75

        plt.axvline(x=spec_line, linewidth=0.5, color="#1e88e5", alpha=alpha_line)
        plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=alpha_text) 

    for e_key, e_val in sl['abs'].items():
        spec_line = float(e_val) * (1+z)
        spec_label = e_key

        plt.axvline(x=spec_line, linewidth=0.5, color="#ff8f00", alpha=0.7)
        plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=0.75)

    # iron spectral lines
    for e_key, e_val in sl['iron'].items(): 
        spec_line = float(e_val) * (1+z)

        plt.axvline(x=spec_line, linewidth=0.5, color="#bdbdbd", alpha=0.3)

    plt.plot(x_data, y_model, linewidth=0.5, color="#b71c1c")

    residuals_mask = (residual > res_stddev) 
    rmask = residuals_mask

    #plt.scatter(x_data[rmask], residual[rmask], s=3, color="#f44336", alpha=0.5)
    plt.scatter(x_data[mask], residual[mask]-1, s=3, color="#43a047")

    #plt.tick_params(labelsize=15)
    plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=15)
    plt.ylabel(r'\textbf{Relative Flux}', fontsize=15)
    plt.savefig("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + str(int(cube_id))
            + "_fitted.pdf")

    plt.close("all")
    
    return {'chi2': total_chi_sq,'redchi2': reduced_chi_sq}

def sigma_sn():
    cubes = np.array([1804])
    to_run = 100 # number of times to run the random generator

    # I want to store every thing which has been generated - what type of array do I 
    # need?

    # 1st dimension: for every cube there is an array
    # 2nd dimension: there are same number of rows as to_run variable
    # 3rd dimension: columns to store data
    #   [0] : new signal value [= signal + perturbation]
    #   [1] : new sigma produced
    #   [2] : (sigma_best - sigma_new) / sigma_best
    #   [3] : new signal to noise value  
    data = np.zeros([len(cubes),to_run,4])
    
    for i_cube in range(len(cubes)):
        cube_id = cubes[i_cube]
        
        # running the best fit fitting routine 
        best_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all")

        z = best_fit['redshift']

        best_noise = best_fit['noise']

        best_x = best_fit['x_data']
        best_y = best_fit['y_data']

        best_variables = best_fit['variables']
        best_sigma = best_variables[1]

        # want to consider between CaH and Hdelta, the range to consider (rtc) is
        #rtc = np.array([3969.588, 4101.89]) * (1+z) 
        rtc = np.array([4000, 4080]) * (1+z) 
        rtc_mask = ((best_x > rtc[0]) & (best_x < rtc[1]))

        best_y_masked = best_y[rtc_mask]
        best_noise_masked = best_noise[rtc_mask]

        noise_median = np.median(best_noise_masked)

        best_sn = np.median(best_y_masked) / noise_median
        
        # the median of the data and the average of the noise should be similar
        n1 = np.std(best_y_masked)
        n2 = np.average(best_noise)
        print(n1, n2)

        original_y = best_fit['y_data_original']
        galaxy_spectrum = original_y 
        
        for i in range(to_run):
            # generating a random noise distribution using a mean of 0 and the 
            # standard deviation of the original galaxy spectrum within a region
            random_noise = np.abs(np.random.normal(0, n1, best_fit['x_length']))
            
            # adding noise to the (new and same) galaxy spectrum
            galaxy_spectrum = galaxy_spectrum + random_noise
            print(galaxy_spectrum)

            print("working with " + str(cube_id) + " and index " + str(i))

            new_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 
                    galaxy_spectrum, "all")

            new_variables = new_fit['variables']
            new_sigma = new_variables[1] 

            sigma_ratio = np.abs(new_sigma - best_sigma) / best_sigma
            
            new_x = new_fit['x_data']
            new_y = new_fit['y_data']
             
            new_mask = ((new_x > rtc[0]) & (new_x < rtc[1]))

            new_x = new_x[new_mask]
            new_y = new_y[new_mask]

            new_model = new_fit['model_data']
            new_signal = np.median(new_y)
            new_noise = np.std(new_y)
 
            print(new_sigma, best_sigma, new_signal/new_noise, new_noise)

            data[i_cube][i][0] = new_signal # new signal
            data[i_cube][i][1] = new_sigma # new sigma
            data[i_cube][i][2] = sigma_ratio # sigma ratio
            data[i_cube][i][3] = new_signal / new_noise # signal to noise
 
    np.save("data/sigma_vs_sn_data", data)

    plt.figure()
    #plt.scatter(best_sn, best_sigma/best_sigma, color="#b71c1c", s=10)
    plt.scatter(data[:,:,3], data[:,:,2], color="#000000", s=10)
    #plt.ylim([np.min(data[:,:,3]), np.max(data[:,:,3])])

    plt.xlabel(r'\textbf{S/N}', fontsize=15)
    plt.ylabel(r'\textbf{$\frac{\Delta \sigma}{\sigma_{best}}$}', fontsize=15)

    plt.tight_layout()
    plt.savefig("graphs/sigma_vs_sn.pdf")
    plt.close("all") 

#chi_squared_cal(1804)
#model_data_overlay(549)
