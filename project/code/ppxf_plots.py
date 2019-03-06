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
    plt.plot(x_data, y_data_scaled+noise_stddev, linewidth=0.1, c="#616161", alpha=0.1)
    plt.plot(x_data, y_data_scaled-noise_stddev, linewidth=0.1, c="#616161", alpha=0.1)
    
    # plotting over the OII doublet
    doublets = np.array([3727.092, 3728.875]) * (1+z)
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
        spec_line = float(e_val)*(1+z)
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
        spec_line = float(e_val)*(1+z)
        spec_label = e_key

        plt.axvline(x=spec_line, linewidth=0.5, color="#ff8f00", alpha=0.7)
        plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=0.75,
                weight="bold", fontsize=15)

    # iron spectral lines
    for e_key, e_val in sl['iron'].items(): 
        spec_line = float(e_val)*(1+z)

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

def linear(x, m, c):
    return (m*x) + c

def chisq(data, data_err, model):
    chisq_indivs = (data-model)**2/data_err**2
    chisq = (chisq_indivs < 100)
    return np.sum(chisq_indivs[chisq])

def sigma_stars_vs_sigma_oii():
    data = np.load("data/ppxf_fitter_data.npy") 

    fig, ax = plt.subplots()

    x_dat = data[:][:,0][:,1]
    y_dat = data[:][:,0][:,2]

    y_mask = (y_dat < 300)

    x_dat = x_dat[y_mask]
    y_dat = y_dat[y_mask]

    xerr=data[:][:,0][:,13][y_mask]
    yerr=data[:][:,0][:,12][y_mask]
   
    ax.errorbar(x_dat, y_dat, xerr=xerr, yerr=yerr,
            color="#000000", fmt="o", ms=5, elinewidth=1.0, 
            capsize=5, capthick=1.0, zorder=0)

    # y/x value
    y_over_x = data[:][:,0][:,2]/data[:][:,0][:,1]
    #ax.annotate("median y/x val: "+str(np.median(y_over_x)), (150,10))

    # plotting 1:1 line
    f_xd = np.linspace(-10,400,500)
    ax.plot(f_xd, f_xd, lw=1.5, color="#000000", alpha=0.3) 

    # Fitting straight-line models to the data y=mx+c
    sl_model = Model(linear) # straight line model

    # Model 1 : free m and c
    md1p = Parameters() # Model 1 parameters
    md1p.add('m', value=1.0)
    md1p.add('c', value=0.0)

    md1r = sl_model.fit(y_dat, x=x_dat, params=md1p) # Model 1 results

    md1_bf = md1r.best_values
    md1_fit = linear(f_xd, md1_bf['m'], md1_bf['c']) 

    # Model 2: constrain m=1.0, free c
    md2p = Parameters()
    md2p.add('m', value=1.0, vary=False)
    md2p.add('c', value=0.0)

    md2r = sl_model.fit(y_dat, x=x_dat, params=md2p) 

    md2_bf = md2r.best_values
    md2_fit = linear(f_xd, md2_bf['m'], md2_bf['c']) 

    # Model 3: free m, constrain c=0.0
    md3p = Parameters()
    md3p.add('m', value=1.0)
    md3p.add('c', value=0.0, vary=False)

    md3r = sl_model.fit(y_dat, x=x_dat, params=md3p) 

    md3_bf = md3r.best_values
    md3_fit = linear(f_xd, md3_bf['m'], md3_bf['c'])

    # Calculating reduced chi-squared values for all three models
    csq_m1 = linear(x_dat, md1_bf['m'], md1_bf['c'])
    rchisq_1 = chisq(y_dat, yerr, csq_m1) / (len(y_dat)-2) # model 1
    
    csq_m2 = linear(x_dat, md2_bf['m'], md2_bf['c'])
    rchisq_2 = chisq(y_dat, yerr, csq_m2) / (len(y_dat)-2) # model 2

    csq_m3 = linear(x_dat, md2_bf['m'], md3_bf['c'])
    rchisq_3 = chisq(y_dat, yerr, csq_m3) / (len(y_dat)-2) # model 3

    print(rchisq_1, rchisq_2, rchisq_3)

    # Plotting best fit lines onto plot

    ax.plot(f_xd, md1_fit, lw=1.5, c="#ce93d8", 
            label=r"\textbf{m: free, c: free, }" + r"$\chi^2_{\nu}$: " + r"$" +
            str(np.round(rchisq_1, decimals=2)) + "$")

    ax.plot(f_xd, md2_fit, lw=1.5, c="#a5d6a7",
            label=r"\textbf{m: fixed, c: free, }" + r"$\chi^2_{\nu}$: " + r"$" +
            str(np.round(rchisq_2, decimals=2)) + "$")

    ax.plot(f_xd, md3_fit, lw=1.5, c="#80deea", 
            label=r"\textbf{m: free, c: fixed, }" + r"$\chi^2_{\nu}$: " + r"$" +
            str(np.round(rchisq_3, decimals=2)) + "$")

 
    """
    low_sn = np.array([554, 765, 849, 1129, 895, 175])
    for i_low in range(len(low_sn)):
        curr_cube = low_sn[i_low] # current cube and it's ID number
        curr_loc = np.where(data[:,:,0][:,0]==curr_cube)[0]
        
        cc_data = data[:][curr_loc,0][0]
        cc_x = cc_data[1]
        cc_y = cc_data[2]

        if cc_y > 300:
            pass
        else:
            ax.scatter(cc_x, cc_y, s=20, c="#d32f2f", zorder=1)
    """

    for i in range(len(data[:][:,0])):
        curr_id = data[:][i,0][0]
        curr_x = data[:][i,0][1]
        curr_y = data[:][i,0][2]

        if curr_y > 300:
            pass
        else:
            ax.annotate(int(curr_id), (curr_x, curr_y))
        
    ax.tick_params(labelsize=20)
    ax.set_ylabel(r'\textbf{$\sigma_{*}$ (kms$^{-1}$)}', fontsize=20)
    ax.set_xlabel(r'\textbf{$\sigma_{OII}$ (kms$^{-1}$)}', fontsize=20)
 
    ax.set_xlim([-10,225]) 
    ax.set_ylim([-10,225])
    #ax.set_aspect('equal', 'box')

    ax.legend(loc='lower right', prop={'size': 12})

    fig.tight_layout()
    fig.savefig("graphs/sigma_star_vs_sigma_oii.pdf",bbox_inches="tight")
    plt.close("all") 

def sigma_ranker():
    data = np.load("data/ppxf_fitter_data.npy") 

    diff_data = np.zeros([len(data[:][:,0]),2])

    array_len = len(data[:][:,0])

    for i_d in range(array_len):
        cc_d = data[:][:,0][i_d] # current cube data
        cube_id = int(cc_d[0])
    
        sigma_oii = cc_d[1]
        sigma_ppxf = cc_d[2]

        # calculating the y-difference between sigma_ppxf and sigma_oii value
        sigma_diff = sigma_ppxf - sigma_oii

        diff_data[i_d][0] = cube_id
        diff_data[i_d][1] = sigma_diff

    diff_data = diff_data[diff_data[:,1].argsort()[::-1]]
    
    fig, axs = plt.subplots(array_len, 3, figsize=(16,16))

    for i_ax in range(array_len):
        cube_id = int(diff_data[i_ax][0])
        cube_diff = float(diff_data[i_ax][1])

        analysis = cube_reader.image_collapser("/Volumes/Jacky_Cao/University/level4/"
                + "project/cubes_better/cube_"+str(cube_id)+".fits")  

        hst_colour = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
                +"_coloured_image_data.npy")

        plt.axis('off')
        axs[i_ax,0].imshow(analysis['median'], cmap='gray_r')
        axs[i_ax,0].set_axis_off()
        axs[i_ax,1].imshow(hst_colour, interpolation='nearest')
        axs[i_ax,1].set_axis_off()
        axs[i_ax,2].annotate(str(cube_id)+", "+str(cube_diff), (0.1,0.5))
        axs[i_ax,2].set_axis_off()

    #fig.tight_layout()
    fig.savefig("graphs/sigma_diff_ranked.pdf")
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

def oii_velocity_old(z):
    c = 299792.458 # speed of light in kms^-1
    num = (z+1)**2 -1 # numerator 
    den = (z+1)**2 + 1# denominator
    return c * (num/den)

def oii_velocity(z):
    c = 299792.458 # speed of light in kms^-1
    return c*np.log(1+z)

def vel_stars_vs_vel_oii():
    data = np.load("data/ppxf_fitter_data.npy") 

    fig, ax = plt.subplots()

    gq_val = [] # graph-quantifier value list

    for i_d in range(len(data[:][:,0])):
        cc_d = data[:][:,0][i_d] # current cube data
        cube_id = int(cc_d[0])
    
        lmfit_vals = spectra_data.lmfit_data(cube_id)
        cube_z = lmfit_vals['z'] 
        cube_z_err = lmfit_vals['err_z']

        vel_oii = oii_velocity(cube_z)
        vel_oii_err = oii_velocity(lmfit_vals['err_z'])
        #vel_oii = 0 # the velocities should be zero as they have not been redshifted

        vel_ppxf = cc_d[14]
        vel_ppxf_err = cc_d[15]

        #print(cube_id, vel_ppxf, vel_oii)

        gq_val.append(vel_ppxf/vel_oii)

        ax.errorbar(vel_oii, vel_ppxf, xerr=vel_oii_err, yerr=vel_ppxf_err, 
                color="#000000", fmt="o", ms=4.5, elinewidth=1.0, 
                capsize=5, capthick=1.0, zorder=0)

        ax.annotate(cube_id, (vel_oii, vel_ppxf))

    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{V$_{*}$ (kms$^{-1}$)}', fontsize=15)
    ax.set_xlabel(r'\textbf{V$_{OII}$ (kms$^{-1}$)}', fontsize=15)

    ax.set_xlim([75000,275000]) 
    ax.set_ylim([75000,275000])

    # plot 1:1 line
    f_xd = np.linspace(0,275000,275000)
    ax.plot(f_xd, f_xd, lw=1.5, color="#000000", alpha=0.3)

    ax.annotate("median y/x val: "+str(np.median(gq_val)), (90_000,260_000))

    fig.tight_layout()
    fig.savefig("graphs/vel_star_vs_vel_oii.pdf")
    plt.close("all") 

def sigma_old_vs_new():
    old_data = np.load("data/ppxf_fitter_data_old.npy")
    new_data = np.load("data/ppxf_fitter_data.npy")

    fig, ax = plt.subplots()

    gq_val = [] # graph-quantifier value list

    for i_d in range(len(old_data[:][:,0])):
        old_ccd = old_data[:][:,0][i_d]
        old_id = int(old_ccd[0])

        new_loc = np.where(new_data[:][:,0][:,0] == old_id)
        new_ccd = new_data[:][:,0][new_loc][0]
    
        old_sigma = old_ccd[2] 
        new_sigma = new_ccd[2]

        gq_val.append(old_sigma/new_sigma)

        ax.errorbar(new_sigma, old_sigma, 
                color="#000000", fmt="o", ms=4.5, elinewidth=1.0, 
                capsize=5, capthick=1.0, zorder=0)
        ax.annotate(str(int(i_d))+"-"+str(old_id), (new_sigma, old_sigma))

    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{$\sigma_{*-old}$ (kms$^{-1}$)}', fontsize=15)
    ax.set_xlabel(r'\textbf{$\sigma_{*-new}$ (kms$^{-1}$)}', fontsize=15)

    ax.set_xlim([0,190]) 
    ax.set_ylim([0,190])

    # plot 1:1 line
    f_xd = np.linspace(0,190,190)
    ax.plot(f_xd, f_xd, lw=1.5, color="#000000", alpha=0.3)

    ax.annotate("median y/x val: "+str(np.median(gq_val)), (10,175))

    fig.tight_layout()
    fig.savefig("graphs/sigma_old_vs_new.pdf")
    plt.close("all") 

def sigma_stars_old_vs_new():
    data = np.load("data/ppxf_fitter_data.npy") 

    old_stars = data[:][:,1][:]
    new_stars = data[:][:,3][:]

    cube_ids = old_stars[:,0]
    old_sigma = old_stars[:,2]
    new_sigma = new_stars[:,2]

    old_sigma_err = old_stars[:,12]
    new_sigma_err = new_stars[:,12]

    gq_val = []

    fig, ax = plt.subplots()

    for i in range(len(cube_ids)):
        cube_id = cube_ids[i]

        cos = old_sigma[i]
        cns = new_sigma[i]
        #print(cos, cns)
    
        cose = old_sigma_err[i]
        cnse = new_sigma_err[i]

        gq_val.append(cns/cos)

        ax.errorbar(cns, cos, xerr=cnse, yerr=cose, 
                color="#000000", fmt="o", ms=4.5, elinewidth=1.0, 
                capsize=5, capthick=1.0, zorder=0)
        ax.annotate(str(int(cube_id)), (cns, cos))

    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{$\sigma_{*-old}$ (kms$^{-1}$)}', fontsize=15)
    ax.set_xlabel(r'\textbf{$\sigma_{*-new}$ (kms$^{-1}$)}', fontsize=15)

    ax.set_xlim([0,600]) 
    ax.set_ylim([0,600])

    ax.set_title(r"\textbf{(3500Å to 4200Å) vs. (4000Å to 4500Å)}")

    # plot 1:1 line
    f_xd = np.linspace(0,750,750)
    ax.plot(f_xd, f_xd, lw=1.5, color="#000000", alpha=0.3)

    ax.annotate("median y/x val: "+str(np.median(gq_val)), (125,10))

    fig.tight_layout()
    fig.savefig("graphs/sigma_stars_old_vs_new.pdf")
    plt.close("all") 



if __name__ == '__main__':
    #chi_squared_cal(1804)
    #model_data_overlay(549)

    #data_graphs()

    #voigt_sigmas()

    sigma_stars_vs_sigma_oii()
    #ranges_sigma_stars_vs_sigma_oii()

    #testing_ranges()

    #vel_stars_vs_vel_oii()
    
    sigma_ranker()

    #sigma_old_vs_new()

    #sigma_stars_old_vs_new()
