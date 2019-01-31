import os
from time import process_time

import io
from contextlib import redirect_stdout

import sys
sys.path.insert(0, '/Users/jackycao/Documents/Projects/scripts/')
import personal_scripts

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
import peakutils

from lmfit import Parameters, Model

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib

from ppxf.ppxf_util import log_rebin

import cube_reader

import ppxf_fitter_kinematics_sdss
import cube_analysis

from os import path

import spectra_data

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
    sl = spectra_data.spectral_lines() 

    # parameters from lmfit
    lm_params = spectra_data.lmfit_data(cube_id)
    c = lm_params['c']
    i1 = lm_params['i1']
    i2 = lm_params['i2']
    sigma_gal = lm_params['sigma_gal']
    z = lm_params['z']
    sigma_inst = lm_params['sigma_inst']

    plt.figure()

    plt.plot(x_data, y_data_scaled, linewidth=1.1, color="#000000")
    plt.plot(x_data, y_data_scaled+noise_stddev, linewidth=0.1, color="#616161", alpha=0.1)
    plt.plot(x_data, y_data_scaled-noise_stddev, linewidth=0.1, color="#616161", alpha=0.1)
    
    # plotting over the OII doublet
    doublets = np.array([3727.092, 3728.875])
    dblt_av = np.average(doublets)

    dblt_x_mask = ((x_data > dblt_av-20) & (x_data < dblt_av+20))
    doublet_x_data = x_data[dblt_x_mask]
    doublet_data = spectra_data.f_doublet(doublet_x_data, c, i1, i2, sigma_gal, z, 
            sigma_inst)
    doublet_data = doublet_data / np.median(y_data)
    plt.plot(doublet_x_data, doublet_data, linewidth=0.5, color="#9c27b0")

    max_y = np.max(y_data_scaled)
    # plotting spectral lines
    for e_key, e_val in sl['emis'].items():
        spec_line = float(e_val)
        spec_label = e_key

        if (e_val in str(doublets)):
            alpha_line = 0.2
        else:
            alpha_line = 0.7
            
        alpha_text = 0.75

        plt.axvline(x=spec_line, linewidth=0.5, color="#1e88e5", alpha=alpha_line)
        plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=alpha_text,
                weight="bold", fontsize=15) 

    for e_key, e_val in sl['abs'].items():
        spec_line = float(e_val)
        spec_label = e_key

        plt.axvline(x=spec_line, linewidth=0.5, color="#ff8f00", alpha=0.7)
        plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=0.75,
                weight="bold", fontsize=15)

    # iron spectral lines
    for e_key, e_val in sl['iron'].items(): 
        spec_line = float(e_val)

        plt.axvline(x=spec_line, linewidth=0.5, color="#bdbdbd", alpha=0.3)

    plt.plot(x_data, y_model, linewidth=1.5, color="#b71c1c")

    residuals_mask = (residual > res_stddev) 
    rmask = residuals_mask

    #plt.scatter(x_data[rmask], residual[rmask], s=3, color="#f44336", alpha=0.5)
    plt.scatter(x_data[mask], residual[mask]-1, s=3, color="#43a047")

    plt.tick_params(labelsize=15)
    plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=15)
    plt.ylabel(r'\textbf{Relative Flux}', fontsize=15)
    plt.tight_layout()
    plt.savefig("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + str(int(cube_id))
            + "_fitted.pdf")

    plt.close("all")
    
    return {'chi2': total_chi_sq,'redchi2': reduced_chi_sq}

def data_graphs():
    data = data = np.load("data/sigma_vs_sn_data.npy")

    # colours list
    colours = [
            "#ab47bc",
            "#43a047",
            "#2196f3",
            "#ff9800"
            ]

    # delta(sigma) vs. pPXF error
    plt.figure()
    for i in range(len(data[:])):
        plt.scatter(data[i][:,2], data[i][:,6], c=colours[i], s=10, alpha=0.2)

    plt.ylabel(r'\textbf{pPXF error}', fontsize=15)
    plt.xlabel(r'\textbf{$\frac{\Delta \sigma}{\sigma_{best}}$}', fontsize=15)

    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("graphs/sigma_vs_ppxf_error.pdf")
    plt.close("all")

def voigt_sigmas():
    data = np.load("data/ppxf_fitter_data.npy")

    fig, ax = plt.subplots()

    ax.scatter(data[:][:,0][:,11], data[:][:,0][:,10], color="#000000", s=10)

    for i in range(len(data[:][:,0])):
        curr_id = data[:][i,0][0]
        curr_x = data[:][i,0][11]
        curr_y = data[:][i,0][10]

        ax.annotate(int(curr_id), (curr_x, curr_y))

    x = np.linspace(0,160,200)
    ax.plot(x, x, lw=1.5, c="#000000", alpha=0.5)

    ax.tick_params(labelsize=15)
    ax.set_xlabel(r'\textbf{Fitted Voigt Sigmas}', fontsize=15)
    ax.set_ylabel(r'\textbf{pPXF Voigt Sigmas}', fontsize=15)

    ax.set_xlim([0,160])
    ax.set_ylim([0,160])

    fig.tight_layout()
    fig.savefig("graphs/voigt_sigmas_1_1.pdf")
    plt.close("all")

def sigma_stars_vs_sigma_oii():
    data = np.load("data/ppxf_fitter_data.npy") 

    fig, ax = plt.subplots()

    yerr=data[:][:,0][:,12]
    xerr=data[:][:,0][:,13]
    ax.errorbar(data[:][:,0][:,1], data[:][:,0][:,2], xerr=xerr, yerr=yerr, 
            color="#000000", fmt="o", ms=4.5, elinewidth=1.0, 
            capsize=5, capthick=1.0, zorder=0)

    low_sn = np.array([554, 765, 849, 1129, 895, 175])

    for i_low in range(len(low_sn)):
        curr_cube = low_sn[i_low] # current cube and it's ID number
        curr_loc = np.where(data[:,:,0][:,0]==curr_cube)[0]
        
        cc_data = data[:][curr_loc,0][0]
        cc_x = cc_data[1]
        cc_y = cc_data[2]
    
        ax.scatter(cc_x, cc_y, s=20, c="#d32f2f", zorder=1)

    for i in range(len(data[:][:,0])):
        curr_id = data[:][i,0][0]
        curr_x = data[:][i,0][1]
        curr_y = data[:][i,0][2]

        ax.annotate(int(curr_id), (curr_x, curr_y))

    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{$\sigma_{*}$ (kms$^{-1}$)}', fontsize=15)
    ax.set_xlabel(r'\textbf{$\sigma_{OII}$ (kms$^{-1}$)}', fontsize=15)

    ax.set_xlim([25,275]) 
    ax.set_ylim([25,275])

    # plot 1:1 line
    f_xd = np.linspace(0,300,300)
    ax.plot(f_xd, f_xd, lw=1.5, color="#000000", alpha=0.3)

    fig.tight_layout()
    fig.savefig("graphs/sigma_star_vs_sigma_oii.pdf")
    plt.close("all") 

def ranges_sigma_stars_vs_sigma_oii():
    # plotting sigma_stars vs. sigma_OII plot for different ranges
    data = np.load("data/ppxf_fitter_data.npy")
    ranges = np.load("data/ppxf_fitting_ranges.npy")

    # in applying different ranges, only the pPXF fitting is affected
    for i_rtc in range(len(ranges)):        
        curr_range = ranges[i_rtc]
        
        d_ci = i_rtc + 1 # current index respective of the data array

        fig, ax = plt.subplots()
        yerr=data[:][:,d_ci][:,12]
        xerr=data[:][:,0][:,13]
        ax.errorbar(data[:][:,0][:,1], data[:][:,d_ci][:,2], xerr=xerr, yerr=yerr, 
                color="#000000", fmt="o", ms=4.5, elinewidth=1.0, 
                capsize=5, capthick=1.0, zorder=0)

        low_sn = np.array([554, 765, 849, 1129, 895, 175])

        for i_low in range(len(low_sn)):
            curr_cube = low_sn[i_low] # current cube and it's ID number
            curr_loc = np.where(data[:,:,0][:,0]==curr_cube)[0]
            
            cc_x = data[:][curr_loc,0][0][1]
            cc_y = data[:][curr_loc,d_ci][0][2]
        
            ax.scatter(cc_x, cc_y, s=20, c="#d32f2f", zorder=1)

        for i in range(len(data[:][:,0])):
            curr_id = data[:][i,0][0]
            curr_x = data[:][i,0][1]
            curr_y = data[:][i,d_ci][2]

            ax.annotate(int(curr_id), (curr_x, curr_y))

        ax.tick_params(labelsize=15)
        ax.set_ylabel(r'\textbf{$\sigma_{*}$ (kms$^{-1}$)}', fontsize=15)
        ax.set_xlabel(r'\textbf{$\sigma_{OII}$ (kms$^{-1}$)}', fontsize=15)

        ax.set_xlim([25,275]) 
        ax.set_ylim([25,275])

        # plot 1:1 line
        f_xd = np.linspace(0,300,300)
        ax.plot(f_xd, f_xd, lw=1.5, color="#000000", alpha=0.3)

        fig.tight_layout()
        range_string = str(curr_range[0]) + "_" + str(curr_range[1])
        fig.savefig("graphs/sigma_star_vs_sigma_oii/"+range_string+".pdf")
        plt.close("all") 

def testing_ranges():
    # plotting sigma_stars vs. sigma_OII plot for different ranges
    data = np.load("data/ppxf_fitter_data.npy")
    ranges = np.load("data/ppxf_fitting_ranges.npy")

    # in applying different ranges, only the pPXF fitting is affected
    for i_rtc in range(len(ranges)):        
        curr_range = ranges[i_rtc]
        
        d_ci = i_rtc + 1 # current index respective of the data array

        fig, ax = plt.subplots()
        ax.scatter(data[:][:,d_ci][:,14], data[:][:,d_ci][:,2], s=10, c="#1e88e5")

        ax.scatter(data[:][:,d_ci][:,14], data[:][:,0][:,2], s=10, c="#000000")

        for i in range(len(data[:][:,0])):
            curr_id = data[:][i,0][0]

            rng_x = data[:][i,d_ci][14]
            acc_x = data[:][i,0][14]

            rng_y = data[:][i,d_ci][2]
            acc_y = data[:][i,0][2]

            ax.annotate(str(int(curr_id))+", "+str(int(i+1)), (rng_x, rng_y))
            ax.annotate(str(int(curr_id))+", "+str(int(i+1)), (acc_x, acc_y))

        ax.tick_params(labelsize=15)
        ax.set_ylabel(r'\textbf{$\sigma_{*}$ (kms$^{-1}$)}', fontsize=15)
        ax.set_xlabel(r'\textbf{Velocity (kms$^{-1}$)}', fontsize=15)

        #ax.set_xlim([25,275]) 
        #ax.set_ylim([25,275])

        # plot 1:1 line
        #f_xd = np.linspace(0,300,300)
        #ax.plot(f_xd, f_xd, lw=1.5, color="#000000", alpha=0.3)

        fig.tight_layout()
        range_string = str(curr_range[0]) + "_" + str(curr_range[1])
        fig.savefig("graphs/testing/"+range_string+".pdf")
        plt.close("all") 

#if __name__ == '__main__':
    #chi_squared_cal(1804)
    #model_data_overlay(549)

    #data_graphs()

    #voigt_sigmas()

    #sigma_stars_vs_sigma_oii()
    #ranges_sigma_stars_vs_sigma_oii()

    #testing_ranges()
