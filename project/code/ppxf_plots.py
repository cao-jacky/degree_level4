import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
import peakutils

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

    # pPXF produces an offset in the data
    pu_peaks = peakutils.indexes(y_data_scaled, thres=4, thres_abs=True)
    pu_peaks_x = peakutils.interpolate(x_data, y_data_scaled, pu_peaks)

    original_peaks_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + 
            str(cube_id) + "_peaks.txt")
    original_peaks_file = open(original_peaks_file)

    opf_line_count = 0 
    for op_crf_line in original_peaks_file:
        if (opf_line_count == 3):
            op_curr_line = op_crf_line.split()
            op_wl = float(op_curr_line[1])
        opf_line_count += 1

    offset_diff = pu_peaks_x[0] - op_wl
    #x_data = x_data - offset_diff

    # using our redshift estimate from lmfit
    cube_result_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) 
            + "_lmfit.txt")
    cube_result_file = open(cube_result_file)

    line_count = 0 
    for crf_line in cube_result_file:
        if (line_count == 20):
            curr_line = crf_line.split()
            z = float(curr_line[1])
        line_count += 1

    plt.figure()

    plt.plot(x_data, y_data_scaled, linewidth=0.1, color="#000000")
    plt.plot(x_data, y_data_scaled+noise_stddev, linewidth=0.1, color="#616161", alpha=0.1)
    plt.plot(x_data, y_data_scaled-noise_stddev, linewidth=0.1, color="#616161", alpha=0.1)
    
    # plotting over the OII doublet
    #doublet_data = 

    max_y = np.max(y_data_scaled)
    # plotting spectral lines
    for e_key, e_val in sl['emis'].items():
        spec_line = float(e_val) * (1+z)
        spec_label = e_key

        plt.axvline(x=spec_line, linewidth=0.5, color="#1e88e5", alpha=0.7)
        plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=0.75) 

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

    plt.scatter(x_data, residual, s=3, color="#f44336", alpha=0.5)
    plt.scatter(x_data[mask], residual[mask], s=3, color="#43a047")

    #plt.tick_params(labelsize=15)
    plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=15)
    plt.ylabel(r'\textbf{Relative Flux}', fontsize=15)
    plt.savefig("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + str(int(cube_id))
            + "_fitted.pdf")

    plt.close("all")
    
    return {'chi2': total_chi_sq,'redchi2': reduced_chi_sq}

def sigma_sn():
    cube = 1804 

#chi_squared_cal(1804)
#model_data_overlay(549)
