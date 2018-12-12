import numpy as np
import matplotlib.pyplot as plt

from scipy.special import wofz

from lmfit import Parameters, Model
from lmfit.models import VoigtModel, ConstantModel

import sys
sys.path.insert(0, '/Users/jackycao/Documents/Projects/scripts/')
import personal_scripts

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

# Voigt profile functions
def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha * np.exp(-(x / alpha)**2 * np.log(2))

def L(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

def V(x, alpha, gamma):
    """ Return the Voigt line shape at x with Lorentzian component HWHM gamma
        and Gaussian component HWHM alpha."""
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

def cah_cak_voigt(x, c, a1, g1, a2, g2):
    return V(x, a1, g1) + V(x, a2, g2) + c

def voigt_fitter(cube_id):
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
        line_count += 1

    # masking out the region of CaH and CaK
    calc_rgn = np.array([3910,4000]) * (1+z)
    #calc_rgn = np.array([3910,3950]) * (1+z)
    
    data_mask = ((data_wl > calc_rgn[0]) & (data_wl < calc_rgn[1]))
    data_wl_masked = data_wl[data_mask]
    data_spec_masked = data_spec[data_mask]

    data_spec_masked = data_spec_masked / np.median(data_spec_masked)
    
    model_mask = ((model_wl > calc_rgn[0]) & (model_wl < calc_rgn[1]))
    model_wl_masked = model_wl[model_mask]
    model_spec_masked = model_spec[model_mask]

    # Applying the lmfit routine to fit two Voigt profiles over our spectra data
    vgt_pars = Parameters()
    vgt_pars.add('v1_amplitude', value=-0.1, max=0.0)
    vgt_pars.add('v1_center', value=3934.777*(1+z), min=3930*(1+z), max=3940*(1+z))
    vgt_pars.add('v1_sigma', value=5, min=0.0)
    vgt_pars.add('v1_gamma', value=0.01)
    vgt_pars.add('c', value=0)
    vgt_pars.add('v2_amplitude', value=-0.1, max=0.0)
    vgt_pars.add('v2_center', value=3969.588*(1+z), min=3950*(1+z), max=3975*(1+z))
    vgt_pars.add('v2_sigma', value=5, min=0.0)
    vgt_pars.add('v2_gamma', value=0.01)

    voigt = VoigtModel(prefix='v1_') + ConstantModel() + VoigtModel(prefix='v2_')

    #vgt_model = Model(voigt)
    vgt_result = voigt.fit(model_spec_masked, x=model_wl_masked, params=vgt_pars)

    opt_pars = vgt_result.best_values
    best_fit = vgt_result.best_fit
    #opt_model = V(data_wl_masked, opt_pars['alpha'], opt_pars['gamma'])

    print(opt_pars)

    # Plotting the spectra
    fig, ax = plt.subplots()
    ax.plot(data_wl_masked, data_spec_masked, lw=1.5, c="#000000")
    ax.plot(model_wl_masked, model_spec_masked, lw=1.5, c="#00c853")
    ax.plot(model_wl_masked, best_fit, lw=1.5, c="#e53935")

    ax.tick_params(labelsize=15)
    ax.set_ylabel(r'\textbf{Flux}', fontsize=15)
    ax.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=15)

    fig.tight_layout()
    fig.savefig("graphs/voigt_fittings/cubes/cube_"+str(cube_id)+"_voigt.pdf")
    plt.close("all")
    

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
