import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

from lmfit import Parameters, Model

from astropy.io import fits

import glob

# Open sky spectrum, fit a Gaussian profile at 6300/6500Å and use the sigma for the 
# instrumental resolution

def gaussian(x, c, i1, mu, sigma):
    norm = (sigma*np.sqrt(2*np.pi))
    term1 = ( i1 / norm ) * np.exp(-(x-mu)**2/(2*sigma**2))
    return (c + term1)

def inst_res():
    sky_spec_file = fits.open("data/skyvariance_csub.fits")
    sky_spec_header = sky_spec_file[0].header
    sky_spec_data = sky_spec_file[0].data

    range_begin = sky_spec_header['CRVAL1']
    pixel_begin = sky_spec_header['CRPIX1']
    step_size = sky_spec_header['CDELT1']
    steps = len(sky_spec_data)
    range_end = range_begin + steps * step_size

    # we have the wavelength range for the data
    wl_range = np.arange(range_begin, range_end, step_size)

    # now we want to fit a Gaussian profile at 6300Å
    sky_gauss_params  = Parameters()
    sky_gauss_params.add('c', value=0)
    sky_gauss_params.add('i1', value=np.max(sky_spec_data), min=0.0)
    sky_gauss_params.add('mu', value=6300, min=6290, max=6310) 
    sky_gauss_params.add('sigma', value=5)

    sky_gauss_model = Model(gaussian)
    sky_gauss_result  = sky_gauss_model.fit(sky_spec_data, x=wl_range, 
            params=sky_gauss_params)

    sky_gauss_vals = sky_gauss_result.best_values
    sky_gauss_fit = sky_gauss_result.best_fit

    sigma_inst = sky_gauss_vals['sigma']
    wl = sky_gauss_vals['mu'] # Units of Å
   
    # We want to find delta_lambda using MUSE spectral resolution of 4200Å and the
    # same wavelength as used in our Gaussian fitting above
    #
    # We want to be using R=lambda/delta_lambda
    R = 4200 # Units of Å
    delta_lambda = wl / R

    print(2.35*sigma_inst, delta_lambda)

    np.save("data/sigma_inst", sigma_inst)

    fig, ax = plt.subplots() 
    ax.plot(wl_range, sky_spec_data, lw=0.5, c="#000000")
    ax.plot(wl_range, sky_gauss_fit, lw=0.5, c="#e53935")

    fig.tight_layout()
    fig.savefig("graphs/testing/sky_gauss.pdf")
    plt.close("all")
   
def synth_spectra_plot():
    cube_id = 1804

    cube_x_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbd_x.npy")
    cube_y_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbs_y.npy")

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

    cube_x_data = cube_x_data / (1+z)

    spec_list = glob.glob('synthe_templates/*.ASC.gz')
    lam_temp = np.loadtxt("/Volumes/Jacky_Cao/University/level4/project/" + 
            "SYNTHE_templates/rp20000/LAMBDA_R20.DAT")
    #spec_y = np.loadtxt(spec_list[0])

    spec1 = np.loadtxt(spec_list[0])
    spec2 = np.loadtxt(spec_list[1])

    fig, ax = plt.subplots() 
    ax.plot(cube_x_data, cube_y_data/np.median(cube_y_data), lw=0.5, c="#e53935")

    for i in range(len(spec_list)):
        spec = np.loadtxt(spec_list[i])
        ax.plot(lam_temp, spec/np.median(spec), lw=0.5, c="#000000")

    fig.tight_layout()

    fig.savefig("graphs/testing/synthe.pdf")
    plt.close("all")


#inst_res()
synth_spectra_plot()
