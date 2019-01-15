import numpy as np
import matplotlib.pyplot as plt

import ppxf_fitter_kinematics_sdss

from lmfit import Parameters, Model
from lmfit.models import VoigtModel, ConstantModel

from scipy.optimize import curve_fit

import datetime

import sys
sys.path.insert(0, '/Users/jackycao/Documents/Projects/scripts/')
import personal_scripts

import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def cauchy(x, x0, g):
    return 1. / ( np.pi * g * ( 1 + ( ( x - x0 )/ g )**2 ) )

def gauss( x, x0, s):
    return 1./ np.sqrt(2 * np.pi * s**2 ) * np.exp( - (x-x0)**2 / ( 2 * s**2 ) )

def voigt1( x, z, s, g, a1 ):
    a = a1
    x0 = 3934.777*(1+z)
    fg = 2 * s * np.sqrt( 2 * np.log(2) )
    fl = 2 * g
    f = ( fg**5 +  2.69269 * fg**4 * fl + 2.42843 * fg**3 * fl**2 + 4.47163 * fg**2 * fl**3 + 0.07842 * fg * fl**4+ fl**5)**(1./5.)
    eta = 1.36603 * ( fl / f ) - 0.47719 * ( fl / f )**2 + 0.11116 * ( f / fl )**3
    return a * ( eta * cauchy( x, x0, f) + ( 1 - eta ) * gauss( x, x0, f ) )

def voigt2( x, z, s, g, a2 ):
    a = a2
    x0 = 3969.588*(1+z)
    fg = 2 * s * np.sqrt( 2 * np.log(2) )
    fl = 2 * g
    f = ( fg**5 +  2.69269 * fg**4 * fl + 2.42843 * fg**3 * fl**2 + 4.47163 * fg**2 * fl**3 + 0.07842 * fg * fl**4+ fl**5)**(1./5.)
    eta = 1.36603 * ( fl / f ) - 0.47719 * ( fl / f )**2 + 0.11116 * ( f / fl )**3
    return a * ( eta * cauchy( x, x0, f) + ( 1 - eta ) * gauss( x, x0, f ) )

def cah_cak(x, z, s, g, a1, a2, c):
    return voigt1(x,z,s,g,a1) + voigt2(x,z,s,g,a2) + c

def voigt_fitter(cube_id):
    # Running pPXF fitting routine
    vars_file = ("ppxf_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))+
            "_ppxf_variables")

    if not (os.path.exists(vars_file)):
        best_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all")
    else:
        best_fit = np.load(vars_file)

    best_fit_vars = best_fit['variables']

    data_wl = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbd_x.npy") # 'x-data'
    data_spec = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbs_y.npy") # 'y-data'

    # y-data which has been reduced down by median during pPXF running
    galaxy = best_fit['y_data'] 

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

    # Applying the lmfit routine to fit two Voigt profiles over our spectra data
    vgt_pars = Parameters()
    vgt_pars.add('sigma_inst', value=sigma_inst, vary=False)
    vgt_pars.add('sigma_gal', value=1.0, min=0.0)

    vgt_pars.add('z', value=z)

    vgt_pars.add('v1_amplitude', value=-0.1)
    vgt_pars.add('v1_center', expr='3934.777*(1+z)')
    vgt_pars.add('v1_sigma', expr='sqrt(sigma_inst**2 + sigma_gal**2)', min=0.0)
    #vgt_pars.add('v1_gamma', value=0.01) # gamma as a parameter is not needed

    vgt_pars.add('v2_amplitude', value=-0.1)
    vgt_pars.add('v2_center', expr='3969.588*(1+z)')
    vgt_pars.add('v2_sigma', expr='v1_sigma')
    #vgt_pars.add('v2_gamma', value=0.01) # gamma is a mirror of sigma apparently 

    vgt_pars.add('c', value=0)

    voigt = VoigtModel(prefix='v1_') + VoigtModel(prefix='v2_') + ConstantModel()
    
    vgt_result = voigt.fit(galaxy_masked, x=z_wl_masked, params=vgt_pars)

    opt_pars = vgt_result.best_values
    best_fit = vgt_result.best_fit

    # Using SciPy as an alternative fitting routine
    popt, pcov = curve_fit(cah_cak, z_wl_masked, galaxy_masked)
    print(popt)

    # Plotting the spectra
    fig, ax = plt.subplots()
    ax.plot(z_wl_masked, galaxy_masked, lw=1.5, c="#000000", alpha=0.3)
    ax.plot(z_wl_masked, model_spec_masked, lw=1.5, c="#00c853")
    ax.plot(z_wl_masked, best_fit, lw=1.5, c="#e53935")

    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{Flux}', fontsize=15)
    ax.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=15)

    fig.tight_layout()
    fig.savefig("graphs/voigt_fittings/cubes/cube_"+str(cube_id)+"_voigt.pdf")
    plt.close("all")

    # obtaining sigmas from pPXF
    sigma_ppxf = best_fit_vars[1]
    sigma_opt = opt_pars['v2_sigma']

    lmfit_report = open("results/voigt/cube_"+str(int(cube_id))+"_lmfit.txt", "w")
    lmfit_report.write("Analysis performed on "+str(datetime.datetime.now())+"\n\n")
    lmfit_report.write("Output from lmfit is the following: \n") 
    lmfit_report.write(vgt_result.fit_report())

    speed_of_light = 299792.458 # speed of light in kms^-1
    sigma_opt_kms = (speed_of_light/sigma_opt) * 10**(-3)

    sigmas = np.array([sigma_ppxf, sigma_opt_kms])

    print(sigma_ppxf, sigma_opt_kms)

    return {'sigmas': sigmas}
    

def example_voigt_plotter():
    alpha, gamma = -0.1, 0.1
    x = np.linspace(-0.8,0.8,1000)

    fig, ax = plt.subplots() 
    ax.plot(x, G(x, alpha), ls=':', label='Gaussian')
    ax.plot(x, L(x, gamma), ls='--', label='Lorentzian')
    ax.plot(x, V(x, alpha, gamma), label='Voigt')
    ax.set_xlim(-0.8,0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig("graphs/voigt_fittings/test_voigt.pdf")
    plt.close("all")

#example_voigt_plotter()
voigt_fitter(1804)
