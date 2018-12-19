import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

from lmfit import Parameters, Model

from astropy.io import fits

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
    

inst_res()
