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
        plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=alpha_text,
                weight="bold", fontsize=15) 

    for e_key, e_val in sl['abs'].items():
        spec_line = float(e_val) * (1+z)
        spec_label = e_key

        plt.axvline(x=spec_line, linewidth=0.5, color="#ff8f00", alpha=0.7)
        plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=0.75,
                weight="bold", fontsize=15)

    # iron spectral lines
    for e_key, e_val in sl['iron'].items(): 
        spec_line = float(e_val) * (1+z)

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

def data_reprocessor():
    data = np.load("data/sigma_vs_sn_data.npy")

    # colours list
    colours = [
            "#f44336",
            "#d81b60",
            "#8e24aa",
            "#5e35b1",
            "#3949ab",
            "#1e88e5",
            "#0097a7",
            "#43a047",
            "#fbc02d",
            "#616161"
            ]

    def sn_vs_delta_sigma_sigma():
        total_bins1 = 400
        X = data[:,:,3] # x-axis should be fractional error
        
        bins = np.linspace(X.min(), X.max(), total_bins1)
        delta = bins[1]-bins[0]
        idx  = np.digitize(X,bins)

        #####
        # S/N vs. delta(sigma)/sigma
        plt.figure()
        for i in range(len(data[:])): 
            plt.scatter(data[i][:,2], data[i][:,3], c=colours[i], s=10, alpha=0.2)
            
        # running median calculator
        Y_sigma = data[:,:,2] # y-axis should be signal to noise
        running_median1 = [np.median(Y_sigma[idx==k]) for k in range(total_bins1)]

        rm1 = np.array(running_median1)        
        y_data1 = (bins-delta/2)

        plt.plot(rm1, y_data1, c="#000000", lw=1.5, alpha=0.7)

        idx = np.isfinite(rm1) # mask to mask out finite values
        fitted_poly = np.poly1d(np.polyfit(rm1[idx], y_data1[idx], 4))
        t = np.linspace(np.min(y_data1), np.max(y_data1), 200)
        plt.plot(fitted_poly(t), t, c="#d32f2f", lw=1.5, alpha=0.8)

        plt.ylabel(r'\textbf{S/N}', fontsize=15)
        plt.xlabel(r'\textbf{$\Delta \sigma / \sigma_{best}$}', fontsize=15)

        #plt.xlim([10**(-4),100])
        #plt.yscale('log')
        plt.tight_layout()
        plt.savefig("graphs/reprocessed_sn_vs_delta_sigma_sigma.pdf")
        plt.close("all") 

        #####
        # S/N vs. delta(sigma_vel)/sigma_vel
        def sn_del_sigma_vel():
            plt.figure()
            for i in range(len(data[:])):
                plt.scatter(data[i][:,5], data[i][:,3], c=colours[i], s=10, alpha=0.2) 

            Y_sigma_vel = data[:,:,5]
            running_median2 = [np.median(Y_sigma_vel[idx==k]) for k in 
                    range(total_bins1)]
            plt.plot(running_median2, bins-delta/2, c="#000000", lw=1.5, alpha=0.7)

            plt.ylabel(r'\textbf{S/N}', fontsize=15)
            plt.xlabel(r'\textbf{$\Delta \sigma_{vel} / \sigma_{vel_{best}}$}', 
                    fontsize=15)

            plt.xlim([-np.min(data[:,:,5]),0.0021])
            #plt.xscale('log')
            plt.tight_layout()
            plt.savefig("graphs/reprocessed_sn_vs_d_sigma_vel_sigma_vel.pdf")
            plt.close("all")

    def sn_vs_delta_sigma():
        # S/N vs. delta(sigma)
        total_bins2 = 400
        X_sigma = data[:,:,3]
        
        bins = np.linspace(X_sigma.min(), X_sigma.max(), total_bins2)
        delta = bins[1]-bins[0]
        idx  = np.digitize(X_sigma,bins)

        plt.figure()
        for i in range(len(data[:])):
            plt.scatter(data[i][:,8], data[i][:,3], c=colours[i], s=10, alpha=0.2)

        Y_sn = data[:,:,8]
        running_median3 = [np.median(Y_sn[idx==k]) for k in range(total_bins2)]
        plt.plot(running_median3, bins-delta/2, c="#000000", lw=1.5, alpha=0.7)

        plt.tick_params(labelsize=15)
        plt.ylabel(r'\textbf{S/N}', fontsize=15)
        plt.xlabel(r'\textbf{$\Delta \sigma$}', fontsize=15)

        #plt.ylim([10**(-8),100])
        #plt.yscale('log')
        plt.tight_layout()
        plt.savefig("graphs/reprocessed_sn_vs_delta_sigma.pdf")
        plt.close("all")

    sn_vs_delta_sigma_sigma()
    sn_vs_delta_sigma()

    os.system('afplay /System/Library/Sounds/Glass.aiff')
    personal_scripts.notifications("ppxf_plots", "Reprocessed plots have been plotted!")

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

def population_gas_sdss(cube_id, tie_balmer, limit_doublets, cube_y, cube_x, 
        noise_array):
    # reading cube_data
    cube_file = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/cube_" 
            + str(cube_id) + ".fits")
    hdu = fits.open(cube_file)
    t = hdu[1].data
 
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

    cube_x_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbd_x.npy")

    cube_x_data = cube_x
    cube_y_data = cube_y

    # masking the data to ignore initial 'noise' / non-features
    initial_mask = (cube_x_data > 3540 * (1+z))
    cube_x_data = cube_x_data[initial_mask] 
    cube_y_data = cube_y_data[initial_mask]

    lamRange = np.array([np.min(cube_x_data), np.max(cube_x_data)]) 
    specNew, logLam, velscale = log_rebin(lamRange, cube_y_data)
    lam = np.exp(logLam)

    # Only use the wavelength range in common between galaxy and stellar library.
    mask = (lam > 6000) & (lam < 7200)
    flux = specNew[mask]
    galaxy = flux/np.median(flux)   # Normalize spectrum to avoid numerical issues
    wave = lam[mask]
 
    # The SDSS wavelengths are in vacuum, while the MILES ones are in air.
    # For a rigorous treatment, the SDSS vacuum wavelengths should be
    # converted into air wavelengths and the spectra should be resampled.
    # To avoid resampling, given that the wavelength dependence of the
    # correction is very weak, I approximate it with a constant factor.
    #
    wave *= np.median(util.vac_to_air(wave)/wave)

    # The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0).
    # A constant noise is not a bad approximation in the fitted wavelength
    # range and reduces the noise in the fit.
    #
    #noise = np.full_like(galaxy, 0.01635)  # Assume constant noise per pixel here

    # cube noise
    #cube_noise_data = ppxf_fitter_kinematics_sdss.cube_noise()
    #spectrum_noise = cube_noise_data['spectrum_noise']
    #spec_noise = spectrum_noise[initial_mask][mask]

    spec_noise = np.abs(noise_array[mask])

    segmentation_data = hdu[2].data
    seg_loc_rows, seg_loc_cols = np.where(segmentation_data == cube_id)
    signal_pixels = len(seg_loc_rows) 

    noise = (spec_noise * np.sqrt(signal_pixels)) / np.median(flux)

    # The velocity step was already chosen by the SDSS pipeline
    # and we convert it below to km/s
    #
    c = 299792.458  # speed of light in km/s
    velscale = c*np.log(wave[1]/wave[0])  # eq.(8) of Cappellari (2017)
    FWHM_gal = 4/6  # SDSS has an approximate instrumental resolution FWHM of 2.76A.
    #FWHM_gal = 0.000001

    #------------------- Setup templates -----------------------
    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
    pathname = ppxf_dir + '/miles_models/Mun1.30*.fits'

    miles = lib.miles(pathname, velscale, FWHM_gal)

    # The stellar templates are reshaped below into a 2-dim array with each
    # spectrum as a column, however we save the original array dimensions,
    # which are needed to specify the regularization dimensions
    #
    reg_dim = miles.templates.shape[1:]
    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

    # See the pPXF documentation for the keyword REGUL,
    regul_err = 0.013  # Desired regularization error

    # Construct a set of Gaussian emission line templates.
    # Estimate the wavelength fitted range in the rest frame.
    #
    lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1 + z)
    gas_templates, gas_names, line_wave = util.emission_lines(
        miles.log_lam_temp, lam_range_gal, FWHM_gal,
        tie_balmer=tie_balmer, limit_doublets=limit_doublets)

    # Combines the stellar and gaseous templates into a single array.
    # During the PPXF fit they will be assigned a different kinematic
    # COMPONENT value
    #
    templates = np.column_stack([stars_templates, gas_templates])

    #-----------------------------------------------------------

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below as described
    # in PPXF_EXAMPLE_KINEMATICS_SAURON and Sec.2.4 of Cappellari (2017)
    #
    c = 299792.458
    dv = c*(miles.log_lam_temp[0] - np.log(wave[0]))  # eq.(8) of Cappellari (2017)
    vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
    start = [vel, 180.]  # (km/s), starting guess for [V, sigma]

    n_temps = stars_templates.shape[1]
    n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
    n_balmer = len(gas_names) - n_forbidden

    # Assign component=0 to the stellar templates, component=1 to the Balmer
    # gas emission lines templates and component=2 to the forbidden lines.
    component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
    gas_component = np.array(component) > 0  # gas_component=True for gas templates

    # Fit (V, sig, h3, h4) moments=4 for the stars
    # and (V, sig) moments=2 for the two gas kinematic components
    moments = [4, 2, 2]

    # Adopt the same starting value for the stars and the two gas components
    start = [start, start, start]

    # If the Balmer lines are tied one should allow for gas reddeining.
    # The gas_reddening can be different from the stellar one, if both are fitted.
    gas_reddening = 0 if tie_balmer else None

    # Here the actual fit starts.
    #
    # IMPORTANT: Ideally one would like not to use any polynomial in the fit
    # as the continuum shape contains important information on the population.
    # Unfortunately this is often not feasible, due to small calibration
    # uncertainties in the spectral shape. To avoid affecting the line strength of
    # the spectral features, we exclude additive polynomials (DEGREE=-1) and only use
    # multiplicative ones (MDEGREE=10). This is only recommended for population, not
    # for kinematic extraction, where additive polynomials are always recommended.
    #
    
    t = process_time()
    f = io.StringIO()
    with redirect_stdout(f):
        pp = ppxf(templates, galaxy, noise, velscale, start,
            plot=False, moments=moments, degree=-1, mdegree=10, vsyst=dv,
            lam=wave, clean=False, regul=1./regul_err, reg_dim=reg_dim,
            component=component, gas_component=gas_component,
            gas_names=gas_names, gas_reddening=gas_reddening)
    
    best_variables = pp.sol
    
    best_fit = pp.bestfit

    # stellar spectrum can be found by performing the following calculation
    #stellar_spectrum = pp.bestfit - pp.gas_bestfit
    
    tied = "_free"
    if ( tie_balmer == True and limit_doublets == True ):
        tied = "_tied"

    gas_populations_file = open("graphs/doublet_testing/cube_" + str(int(cube_id)) + 
            "_gas_population" + tied + ".txt", 'w')

    gas_populations_file.write(f.getvalue())
    gas_populations_file.write("")

    # When the two Delta Chi^2 below are the same, the solution
    # is the smoothest consistent with the observed spectrum.
    #
    gas_populations_file.write(('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size) 
        + "\n"))
    gas_populations_file.write(('Current Delta Chi^2: %.4g' % 
        ((pp.chi2 - 1)*galaxy.size) + "\n"))
    gas_populations_file.write(('Elapsed time in PPXF: %.2f s' % (process_time() - t)
        + "\n\n"))

    weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
    weights = weights.reshape(reg_dim)/weights.sum()  # Normalized

    g = io.StringIO()
    with redirect_stdout(g):
        miles.mean_age_metal(weights)
        miles.mass_to_light(weights, band="r")

    gas_populations_file.write(g.getvalue())

    # Plot fit results for stars and gas.
    plt.clf()
    plt.subplot(211)
    pp.plot()

    # Plot stellar population mass fraction distribution
    plt.subplot(212)
    miles.plot(weights)
    plt.tight_layout()
    #plt.pause(1)
    #plt.show()
 
    gas_populations_graph = ("graphs/doublet_testing/cube_" + str(int(cube_id)) + 
            "_gas_populations" + tied + ".pdf")
    plt.savefig(gas_populations_graph)
    plt.close("all")

    return {'variables': best_variables, 'spectrum_fit': best_fit, 'x_axis': wave, 
            'og_y': galaxy}

def oii_plots():
    data = np.load("data/ppxf_fitter_data.npy")
    
    for i in range(len(data[:][:,0])):
        cube_id = int(data[:][i,0][0])

        sigma_lmfit = data[:][i,0][1] # lmfit
        sigma_ppxf = data[:][i,0][5] # ppxf

        c = 299792.458 # speed of light in kms^-1
        sigma_lmfit = (c/sigma_lmfit) * 10**(-3)
        sigma_ppxf = (c/sigma_ppxf) * 10**(-3)

        print(sigma_lmfit, sigma_ppxf)

        # variables for the Gaussian doublet from lmfit
        cube_result_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + 
                str(cube_id) + "_lmfit.txt")
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
        
        x = np.linspace(3500, 4000, 500) * (1+z)

        fit_lmfit = f_doublet(x, c, i1, i2, sigma_lmfit, z, sigma_inst)
        fit_ppxf = f_doublet(x, c, i1, i2, sigma_ppxf, z, sigma_inst)

        fig, ax = plt.subplots()

        ax.plot(x, fit_lmfit, color="#000000", lw=1.5)
        ax.plot(x, fit_ppxf, color="#e53935", lw=1.5)

        ax.tick_params(labelsize=15)
        ax.set_ylabel(r'\textbf{Flux}', fontsize=15)
        ax.set_xlabel(r'\textbf{Wavelength \AA}', fontsize=15)

        fig.tight_layout()
        fig.savefig("graphs/doublets/cube_"+str(cube_id)+".pdf")
        plt.close("all") 

def sigma_ranges(sigma_input):
    cube_id = 1804

    # variables for the Gaussian doublet from lmfit
    cube_result_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + 
            str(cube_id) + "_lmfit.txt")
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

    #sigma_inst = 0 

    cube_x_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbd_x.npy") 
    cube_y_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbs_y.npy")

    x = cube_x_data

    speed_of_light = 299792.458 # speed of light in kms^-1

    sigma_input = (speed_of_light/sigma_input) * 10**(-3)

    example_doublet = f_doublet(x, c, i1, i2, sigma_input, z, sigma_inst)

    # now I want to fit a Gaussian over this
    gss_pars = Parameters()
    gss_pars.add('c', value=c)
    gss_pars.add('i1', value=i1, min=0.0)
    gss_pars.add('r', value=1.3, min=0.5, max=1.5)
    gss_pars.add('i2', expr='i1/r', min=0.0)
    gss_pars.add('sigma_gal', value=sigma_input)
    gss_pars.add('z', value=z)
    gss_pars.add('sigma_inst', value=sigma_inst, vary=False)

    gss_model = Model(f_doublet)
    gss_result = gss_model.fit(example_doublet, x=x, params=gss_pars)

    opti_pms = gss_result.best_values

    lmfit_gauss = f_doublet(x, opti_pms['c'], opti_pms['i1'], opti_pms['i2'], 
            opti_pms['sigma_gal'], opti_pms['z'], opti_pms['sigma_inst'])

    doublet_range = ((cube_x_data > 3600*(1+z)) & (cube_x_data < 3750*(1+z)))
   
    dlt_min = 3720*(1+z)
    dlt_max = 3740*(1+z)

    range_begin = np.where(cube_x_data < dlt_min)[0][-1]
    range_end = np.where(cube_x_data < dlt_max)[0][-1]

    cube_y_data[range_begin:range_end] = example_doublet[range_begin:range_end]

    gas_fit = population_gas_sdss(1804, tie_balmer=False, limit_doublets=False, 
            cube_y=cube_y_data)

    sigma_ppxf = gas_fit['variables']
    #sigma_ppxf1 = speed_of_light / (sigma_ppxf[1][1] * 10**(3)) 
    #sigma_ppxf2 = speed_of_light / (sigma_ppxf[2][1] * 10**(3)) 
   
    sigma_ppxf1 = sigma_ppxf[1][1]
    sigma_ppxf2 = sigma_ppxf[2][1]

    print(opti_pms)
    print(sigma_input,sigma_inst, sigma_ppxf1, sigma_ppxf2)

    fig, ax = plt.subplots()

    ax.plot(cube_x_data, cube_y_data, color="#000000", lw=0.5)
    
    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{Flux}', fontsize=15)
    ax.set_xlabel(r'\textbf{Wavelength \AA}', fontsize=15)

    fig.tight_layout()
    fig.savefig("graphs/doublets/test_cube_"+str(cube_id)+"_"+str(sigma_input)+
            "_spectra.pdf")
    plt.close("all") 

    fig, ax = plt.subplots()

    ax.plot(x, example_doublet, color="#000000", lw=0.5)
    ax.plot(x, lmfit_gauss, color="#f44336", lw=0.5)
    
    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{Flux}', fontsize=15)
    ax.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=15)

    ax.set_xlim([3720*(1+z),3740*(1+z)])

    fig.tight_layout()
    fig.savefig("graphs/doublets/test_cube_"+str(cube_id)+"_"+str(sigma_input)+
            ".pdf")
    plt.close("all") 

    return gas_fit['variables'], opti_pms

def oii_doublet_testing():  
    x = np.arange(50,300,10)
    sigma_data = np.zeros([len(x), 4])
    for i in range(len(x)):
        stc = x[i] # sigma to consider
        processed = sigma_ranges(stc)

        ppxf_vars = processed[0]
        
        sigma_ppxf1 = ppxf_vars[1][1]
        sigma_ppxf2 = ppxf_vars[2][1]

        gas_vars = processed[1]

        sigma_gas = gas_vars['sigma_gal']

        sigma_data[i][0] = stc
        sigma_data[i][1] = sigma_ppxf1
        sigma_data[i][2] = sigma_ppxf2
        sigma_data[i][3] = sigma_gas

    print(sigma_data)

    fig, ax = plt.subplots()

    x_axis = np.sqrt(sigma_data[:,1]**2 - sigma_data[:,2]**2)
    ax.scatter(x_axis, sigma_data[:,0], color="#000000", s=10)
    
    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{Sigma Input}', fontsize=15)
    ax.set_xlabel(r'\textbf{Sigma Output}', fontsize=15) 

    cube_id = 1804
    # variables for the Gaussian doublet from lmfit
    cube_result_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + 
            str(cube_id) + "_lmfit.txt")
    cube_result_file = open(cube_result_file)

    line_count = 0 
    for crf_line in cube_result_file:
        if (line_count == 20):
            curr_line = crf_line.split()
            z = float(curr_line[1])
        line_count += 1

    fig.tight_layout()
    fig.savefig("graphs/doublets/sigma_input_vs_sigma_output.pdf")
    plt.close("all")

def custom_oii_testing():
    sigma_tc = np.arange(50,500,10)
    sigma_data = np.zeros([len(sigma_tc), 4])

    for i in range(len(sigma_tc)):
        # let's say we want a spectrum is from 6000Å to 7000Å
        #spectrum = np.zeros([2000])  
        x_range = np.linspace(6000,7000,2000) # the spacing is 0.5Å 

        cube_id = 1804

        # variables for the Gaussian doublet from lmfit
        cube_result_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + 
                str(cube_id) + "_lmfit.txt")
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

        speed_of_light = 299792.458 # speed of light in kms^-1

        #sigma_input = 500
        stc = sigma_tc[i] # sigma to consider
        sigma_input = stc
        sigma_input = (speed_of_light/sigma_input) * 10**(-3)

        sigma_input_save = (speed_of_light/sigma_input) * 10**(-3)

        # creating a spectra which runs from 6000Å to 7000Å - 
        y_spectra = f_doublet(x_range, c, i1, i2, sigma_input, z, sigma_inst)

        fig, ax = plt.subplots()

        ax.plot(x_range, y_spectra, color="#000000", lw=0.5)
        
        ax.tick_params(labelsize=15)
        ax.set_ylabel(r'\textbf{Flux}', fontsize=15)
        ax.set_xlabel(r'\textbf{Wavelength \AA}', fontsize=15)

        fig.tight_layout()
        fig.savefig("graphs/doublet_testing/test_cube_"+str(cube_id)+"_"+
                str(sigma_input_save)+"_no_noise_spectra.pdf")
        plt.close("all")

        # adding noise to that spectrum
        np.random.seed(0)
        noise = np.random.normal(0,100,2000) 

        y_spectra = y_spectra + noise
       
        gas_fit = population_gas_sdss(1804, tie_balmer=True, limit_doublets=True, 
                cube_y=y_spectra, cube_x=x_range, noise_array=noise)

        fig, ax = plt.subplots()

        ax.plot(x_range, y_spectra, color="#000000", lw=0.5)
        
        ax.tick_params(labelsize=15)
        ax.set_ylabel(r'\textbf{Flux}', fontsize=15)
        ax.set_xlabel(r'\textbf{Wavelength \AA}', fontsize=15)

        fig.tight_layout()
        fig.savefig("graphs/doublet_testing/test_cube_"+str(cube_id)+"_"+
                str(sigma_input_save)+"_spectra.pdf")
        plt.close("all") 
                
        ppxf_vars = gas_fit['variables']
        
        sigma_ppxf1 = ppxf_vars[1][1]
        sigma_ppxf2 = ppxf_vars[2][1]

        sigma_data[i][0] = stc
        sigma_data[i][1] = sigma_ppxf1
        sigma_data[i][2] = sigma_ppxf2

        ppxf_fit = gas_fit['spectrum_fit'] 
        ppxf_x = gas_fit['x_axis']

        ppxf_og_y = gas_fit['og_y']

        # I could refit the spectra onto the final pPXF fitting in a round fashion...
        # now I want to fit a Gaussian over this
        gss_pars = Parameters()
        gss_pars.add('c', value=c)
        gss_pars.add('i1', value=i1, min=0.0)
        gss_pars.add('r', value=1.3, min=0.5, max=1.5)
        gss_pars.add('i2', expr='i1/r', min=0.0)
        gss_pars.add('sigma_gal', value=sigma_input)
        gss_pars.add('z', value=z)
        gss_pars.add('sigma_inst', value=sigma_inst, vary=False)

        gss_model = Model(f_doublet)
        gss_result = gss_model.fit(ppxf_fit, x=ppxf_x, params=gss_pars)

        opti_pms = gss_result.best_values

        ppxf_refit = f_doublet(ppxf_x, opti_pms['c'], opti_pms['i1'], opti_pms['i2'], 
                opti_pms['sigma_gal'], opti_pms['z'], opti_pms['sigma_inst'])

        fig, ax = plt.subplots()

        ax.plot(ppxf_x, ppxf_og_y, color="#000000", lw=1.5, alpha=0.2)
        ax.plot(ppxf_x, ppxf_fit, color="#e53935", lw=1.5, alpha=0.7, linestyle='-')
        ax.plot(ppxf_x, ppxf_refit, c="#00c853", lw=1.5, alpha=0.7)

        ax.tick_params(labelsize=15)
        ax.set_ylabel(r'\textbf{Flux}', fontsize=15)
        ax.set_xlabel(r'\textbf{Wavelength \AA}', fontsize=15)

        fig.tight_layout() 
        fig.savefig("graphs/doublet_testing/overlays/cube_"+str(cube_id)+"_"+
                str(sigma_input_save)+"_spectra.pdf")
        plt.close("all")

        sigma_data[i][3] = (speed_of_light/opti_pms['sigma_gal']) * 10**(-3)

    print(sigma_data)

    np.save("cube_results/sigma_data", sigma_data)

    fig, ax = plt.subplots()

    x_axis = np.sqrt(sigma_data[:,1]**2 - sigma_data[:,2]**2)
    ax.scatter(x_axis, sigma_data[:,0], color="#000000", s=10)
    
    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{Sigma Input}', fontsize=15)
    ax.set_xlabel(r'\textbf{Sigma Output}', fontsize=15) 

    fig.tight_layout()
    fig.savefig("graphs/doublet_testing/sigma_input_vs_sigma_output.pdf")
    plt.close("all")

def oii_doublet_plotter():
    sigma_data = np.load("cube_results/sigma_data.npy")

    fig, ax = plt.subplots()

    #x_axis = np.sqrt(sigma_data[:,1]**2 - sigma_data[:,2]**2)
    ax.scatter(sigma_data[:,3], sigma_data[:,0], color="#000000", s=10)

    s_line = np.linspace(0,500,500)# straight line

    ax.plot(s_line, s_line, lw=1.5, c="#000000")
    
    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{Sigma Input}', fontsize=15)
    ax.set_xlabel(r'\textbf{Sigma Output}', fontsize=15)

    ax.set_xlim([0,500])
    ax.set_ylim([0,500])

    fig.tight_layout()
    fig.savefig("graphs/doublet_testing/sigma_input_vs_sigma_output_1_1_tied.pdf")
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

#chi_squared_cal(1804)
#model_data_overlay(549)

#data_reprocessor()
#data_graphs()

#oii_plots()
#sigma_ranges(250)
#oii_doublet_testing()

#custom_oii_testing()
#oii_doublet_plotter()

#voigt_sigmas()
