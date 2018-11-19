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

def chi_squared_cal(cube_id):
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
                'O[II]':    '3727',
                'CaK':      '3933',
                'CaH':      '3968',
                'Hdelta':   '4101', 
                }, 
            'abs': {'K': '3934.777',
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

    ## plotting spectra lines
    for e_key, e_val in sl['emis'].items():
        spec_line = float(e_val) * (1+z)
        plt.axvline(x=spec_line, linewidth=0.5, color="#00c853")
        #plt.text(spec_line-10, 4800, e_key, rotation=-90)

    plt.plot(x_data, y_model, linewidth=0.5, color="#b71c1c")
    #plt.plot(x_data, noise, linewidth=0.5, color="#8e24aa")

    plt.axhline(res_stddev, linewidth=0.5, color="#000000", alpha=0.3)
    plt.axhline(res_median, linewidth=0.5, color="#000000", alpha=0.3)
    plt.axhline(-res_stddev, linewidth=0.5, color="#000000", alpha=0.3)

    plt.scatter(x_data, residual, s=3, color="#f44336")
    plt.scatter(x_data[mask], residual[mask], s=3, color="#43a047")

    #plt.tick_params(labelsize=15)
    #plt.title(r'\textbf{'+str(reduced_chi_sq)+'}', fontsize=15)
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
