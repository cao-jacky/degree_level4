from time import process_time
from os import path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib

from ppxf.ppxf_util import log_rebin

import cube_reader

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

##############################################################################

def population_gas_sdss(cube_id, tie_balmer, limit_doublets):

    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__)) 

    # reading cube_data
    cube_file = "data/cubes/cube_" + str(cube_id) + ".fits"
    hdu = fits.open(cube_file)
    t = hdu[0].data
 
    # using our redshift estimate from lmfit
    cube_result_file = ("results/cube_" + str(cube_id) + "/cube_" + str(cube_id) + 
            "_lmfit.txt")
    cube_result_file = open(cube_result_file)

    line_count = 0 
    for crf_line in cube_result_file:
        if (line_count == 20):
            curr_line = crf_line.split()
            z = float(curr_line[1])
        line_count += 1

    cube_x_data = np.load("results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbd_x.npy")
    cube_y_data = np.load("results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbs_y.npy")

    # Only use the wavelength range in common between galaxy and stellar library.
    #
    mask = (cube_x_data > 6000) & (cube_x_data < 7200)
    flux = cube_y_data[mask]
    galaxy = flux/np.median(flux)   # Normalize spectrum to avoid numerical issues
    wave = cube_x_data[mask]

    sky_data = cube_reader.sky_noise("data/skyvariance_csub.fits") 
    sky_noise = sky_data[mask]
 
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
    noise = np.full_like(galaxy, 0.01635)  # Assume constant noise per pixel here

    # The velocity step was already chosen by the SDSS pipeline
    # and we convert it below to km/s
    #
    c = 299792.458  # speed of light in km/s
    velscale = c*np.log(wave[1]/wave[0])  # eq.(8) of Cappellari (2017)
    FWHM_gal = 2.76  # SDSS has an approximate instrumental resolution FWHM of 2.76A.

    #------------------- Setup templates -----------------------

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
    pp = ppxf(templates, galaxy, noise, velscale, start,
              plot=False, moments=moments, degree=-1, mdegree=10, vsyst=dv,
              lam=wave, clean=False, regul=1./regul_err, reg_dim=reg_dim,
              component=component, gas_component=gas_component,
              gas_names=gas_names, gas_reddening=gas_reddening)

    # When the two Delta Chi^2 below are the same, the solution
    # is the smoothest consistent with the observed spectrum.
    #
    print('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size))
    print('Current Delta Chi^2: %.4g' % ((pp.chi2 - 1)*galaxy.size))
    print('Elapsed time in PPXF: %.2f s' % (process_time() - t))

    weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
    weights = weights.reshape(reg_dim)/weights.sum()  # Normalized

    miles.mean_age_metal(weights)
    miles.mass_to_light(weights, band="r")

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

##############################################################################

#if __name__ == '__main__':

    #print("\n===============================================\n" +
             #" Fit with free Balmer lines and [SII] doublet: \n" +
             #"===============================================\n")

    #population_gas_sdss(cube_id=23,tie_balmer=False, limit_doublets=False)

    #print("\n=======================================================\n" +
             #" Fit with tied Balmer lines and limited [SII] doublet: \n" +
             #"=======================================================\n")

    # Note tha the inclusion of a few extra faint Balmer lines is sufficient to
    # decrease the chi2 of the fit, even though the Balmer decrement is fixed.
    # In this case, the best-fitting gas reddening is at the E(B-V)=0 boundary.
    #ppxf_example_population_gas_sdss(tie_balmer=True, limit_doublets=True)
