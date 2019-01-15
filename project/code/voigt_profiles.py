import numpy as np
import matplotlib.pyplot as plt

import ppxf_fitter_kinematics_sdss

from lmfit import Parameters, Model
from lmfit.models import VoigtModel, ConstantModel

import peakutils

import datetime

import sys
sys.path.insert(0, '/Users/jackycao/Documents/Projects/scripts/')
import personal_scripts

import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def voigt_fitter(cube_id):
    # Running pPXF fitting routine
    vars_file = ("ppxf_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))+
            "_ppxf_variables.npy")
    if not (os.path.exists(vars_file)):
        best_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all")
        best_fit_vars = best_fit['variables']
    else:
        best_fit_vars = np.load(vars_file)

    # y-data which has been reduced down by median during pPXF running
    gal_file = ("ppxf_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))+
            "_galaxy.npy")
    if not (os.path.exists(gal_file)):
        best_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all")
        galaxy = best_fit['y_data']
    else:
        galaxy = np.load(gal_file)

    data_wl = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbd_x.npy") # 'x-data'
    data_spec = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbs_y.npy") # 'y-data' 

    model_wl = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_lamgal.npy") 
    model_spec = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_model.npy")

    cube_result_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) 
            + "_lmfit.txt")
    cube_result_file = open(cube_result_file)

    line_count = 0 
    for crf_line in cube_result_file:
        if (line_count == 20):
            curr_line = crf_line.split()
            z = float(curr_line[1])
        if (line_count == 21):
            curr_line = crf_line.split()
            sigma_inst = float(curr_line[1])
        line_count += 1

    # masking out the region of CaH and CaK
    calc_rgn = np.array([3900,4000]) 
    
    data_rgn = calc_rgn * (1+z)
    data_mask = ((data_wl > data_rgn[0]) & (data_wl < data_rgn[1]))
    data_wl_masked = data_wl[data_mask]
    data_spec_masked = data_spec[data_mask]

    data_spec_masked = data_spec_masked / np.median(data_spec_masked)
    
    model_rgn = calc_rgn
    model_mask = ((model_wl > calc_rgn[0]) & (model_wl < calc_rgn[1]))
    model_wl_masked = model_wl[model_mask]
    model_spec_masked = model_spec[model_mask]

    z_wl_masked = model_wl_masked * (1+z) # redshifted wavelength range
    galaxy_masked = galaxy[model_mask]

    bl_guess = np.median(galaxy_masked) # base-line guess
    voigt_min = np.min(galaxy_masked) - bl_guess

    sigma_ppxf_kms = best_fit_vars[1] 

    speed_of_light = 299792.458 # speed of light in kms^-1
    sigma_ppxf = speed_of_light / (sigma_ppxf_kms * 10**(3))

    # Applying the lmfit routine to fit two Voigt profiles over our spectra data
    vgt_pars = Parameters()
    vgt_pars.add('sigma_inst', value=sigma_inst, vary=False)
    vgt_pars.add('sigma_gal', value=sigma_ppxf, min=0.0)

    vgt_pars.add('z', value=z)

    vgt_pars.add('v1_amplitude', value=voigt_min+1)
    vgt_pars.add('v1_center', expr='3934.777*(1+z)')
    vgt_pars.add('v1_sigma', expr='sqrt(sigma_inst**2 + sigma_gal**2)', min=0.0)
    #vgt_pars.add('v1_gamma', value=0.01) # gamma as a parameter is not needed

    vgt_pars.add('v2_amplitude', value=voigt_min)
    vgt_pars.add('v2_center', expr='3969.588*(1+z)')
    vgt_pars.add('v2_sigma', expr='v1_sigma')
    #vgt_pars.add('v2_gamma', value=0.01) # gamma is a mirror of sigma apparently 

    vgt_pars.add('c', value=bl_guess)

    voigt = VoigtModel(prefix='v1_') + VoigtModel(prefix='v2_') + ConstantModel()
    
    init = voigt.eval(vgt_pars, x=z_wl_masked)

    vgt_result = voigt.fit(galaxy_masked, x=z_wl_masked, params=vgt_pars)
    opt_pars = vgt_result.best_values
    best_fit = vgt_result.best_fit

    # Plotting the spectra
    fig, ax = plt.subplots()
    ax.plot(z_wl_masked, galaxy_masked, lw=1.5, c="#000000", alpha=0.3)
    ax.plot(z_wl_masked, model_spec_masked, lw=1.5, c="#00c853")
    ax.plot(z_wl_masked, best_fit, lw=1.5, c="#e53935")
    ax.plot(z_wl_masked, init, 'k--', lw=1.5, alpha=0.2)

    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{Flux}', fontsize=15)
    ax.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=15)

    fig.tight_layout()
    fig.savefig("graphs/voigt_fittings/cubes/cube_"+str(cube_id)+"_voigt.pdf")
    plt.close("all")

    lmfit_report = open("results/voigt/cube_"+str(int(cube_id))+"_lmfit.txt", "w")
    lmfit_report.write("Analysis performed on "+str(datetime.datetime.now())+"\n\n")
    lmfit_report.write("Output from lmfit is the following: \n") 
    lmfit_report.write(vgt_result.fit_report())

    # obtaining sigmas from pPXF
    sigma_ppxf = best_fit_vars[1]
    sigma_opt = opt_pars['v2_sigma']

    speed_of_light = 299792.458 # speed of light in kms^-1
    sigma_opt_kms = (speed_of_light/sigma_opt) * 10**(-3)

    sigmas = np.array([sigma_ppxf, sigma_opt_kms])

    print(sigma_ppxf, sigma_opt_kms)

    return {'sigmas': sigmas}
    
voigt_fitter(1804)
