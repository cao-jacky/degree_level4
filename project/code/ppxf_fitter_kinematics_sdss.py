import glob
from time import process_time 
import os
from os import path

import io
from contextlib import redirect_stdout

from astropy.io import fits
import numpy as np

import peakutils

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

from ppxf.ppxf_util import log_rebin
import matplotlib.pyplot as plt

import cube_reader
import cube_analysis

sky_noise = cube_reader.sky_noise("data/skyvariance_csub.fits")

def cube_noise():
    cube_noise_file = "data/cube_noise_std.fits"
    cube_noise_file = fits.open(cube_noise_file)

    cube_noise_data = cube_noise_file[1].data 
    noise = np.sum(np.abs(cube_noise_data))
    return {'noise_value': noise, 'spectrum_noise': cube_noise_data}

def kinematics_sdss(cube_id, perturbation):
    file_loc = "ppxf_results" + "/cube_" + str(int(cube_id))
    if not os.path.exists(file_loc):
        os.mkdir(file_loc) 

    # reading cube_data
    cube_file = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/cube_"
        + str(cube_id) + ".fits")
    hdu = fits.open(cube_file)
    t = hdu[1].data

    spectra = cube_reader.spectrum_creator(cube_file)
     
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
    cube_y_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbs_y.npy") 

    # applying the perturbation to the y data
    cube_y_data = cube_y_data + perturbation

    initial_mask = (cube_x_data > 3540 * (1+z))
    cube_x_data = cube_x_data[initial_mask] 
    cube_y_data = cube_y_data[initial_mask]

    lamRange = np.array([np.min(cube_x_data), np.max(cube_x_data)]) 
    specNew, logLam, velscale = log_rebin(lamRange, cube_y_data)
    lam = np.exp(logLam)
    
    loglam = np.log10(lam)
    # Only use the wavelength range in common between galaxy and stellar library.
    mask = (loglam > np.log10(3540)) & (loglam < np.log10(7409))
    flux = specNew[mask]
 
    galaxy = flux/np.median(flux)   # Normalize spectrum to avoid numerical issues
    loglam_gal = loglam[mask]
    lam_gal = 10**loglam_gal 

    cube_noise_data = cube_noise()
    spectrum_noise = cube_noise_data['spectrum_noise']
    spec_noise = spectrum_noise[initial_mask][mask]

    segmentation_data = hdu[2].data
    seg_loc_rows, seg_loc_cols = np.where(segmentation_data == cube_id)
    signal_pixels = len(seg_loc_rows)

    noise = spec_noise * np.sqrt(signal_pixels) / np.median(flux)

    c = 299792.458                  # speed of light in km/s
    frac = lam_gal[1]/lam_gal[0]    # Constant lambda fraction per pixel
    dlam_gal = (frac - 1)*lam_gal   # Size of every pixel in Angstrom

    data_shape = np.shape(galaxy)
    wdisp = np.full(data_shape, 1, dtype=float) # Intrinsic dispersion of every pixel

    fwhm_gal = 2.51*wdisp*dlam_gal # Resolution FWHM of every pixel, in Angstroms
    velscale = np.log(frac)*c       # Constant velocity scale in km/s per pixel

    # If the galaxy is at significant redshift, one should bring the galaxy
    # spectrum roughly to the rest-frame wavelength, before calling pPXF
    # (See Sec2.4 of Cappellari 2017). In practice there is no
    # need to modify the spectrum in any way, given that a red shift
    # corresponds to a linear shift of the log-rebinned spectrum.
    # One just needs to compute the wavelength range in the rest-frame
    # and adjust the instrumental resolution of the galaxy observations.
    # This is done with the following three commented lines:
    #
    #lam_gal = lam_gal/(1+z)  # Compute approximate restframe wavelength
    #fwhm_gal = fwhm_gal/(1+z)   # Adjust resolution in Angstrom

    # Read the list of filenames from the Single Stellar Population library
    # by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
    # of the library is included for this example with permission
    #template_set = glob.glob('miles_models/Mun1.30Z*.fits')
    template_set = glob.glob('jacoby_models/jhc0*.fits')
    fwhm_tem = 4.5 # instrumental resolution in Ångstroms.

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the SDSS galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = fits.open(template_set[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
    lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
    sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
    templates = np.empty((sspNew.size, len(template_set)))

    # Interpolates the galaxy spectral resolution at the location of every pixel
    # of the templates. Outside the range of the galaxy spectrum the resolution
    # will be extrapolated, but this is irrelevant as those pixels cannot be
    # used in the fit anyway.
    fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SDSS and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SDSS
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    # In the line below, the fwhm_dif is set to zero when fwhm_gal < fwhm_tem.
    # In principle it should never happen and a higher resolution template should be used.
    #
    fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
    sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

    for j, fname in enumerate(template_set):
        hdu = fits.open(fname)
        ssp = hdu[0].data
        ssp = util.gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
        sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
        templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below (see above).
    #
    c = 299792.458
    dv = np.log(lam_temp[0]/lam_gal[0])*c    # km/s

    #lam_gal_alt = lam_gal * (1+z)
    #lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
    goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, z)
    
    # sky
    skyNew, skyLogLam, skyVelScale = log_rebin(lamRange, sky_noise)
    skyNew = skyNew[initial_mask][mask]

    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #
    vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
    t = process_time()

    f = io.StringIO()
    with redirect_stdout(f):
        pp = ppxf(templates, galaxy, noise, velscale, start, sky=skyNew,
            goodpixels=goodpixels, plot=True, moments=4,
            degree=12, vsyst=dv, clean=False, lam=lam_gal) 

    ppxf_variables = pp.sol

    if (perturbation == 0):
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_lamgal", lam_gal) 
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_flux", flux)

        x_data = cube_x_data[mask]
        y_data = cube_y_data[mask]
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_x", x_data)
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_y", y_data)

        # if best fit i.e. perturbation is 0, save everything
     
        kinematics_file = open(file_loc + "/cube_" + str(int(cube_id)) + 
            "_kinematics.txt", 'w')

        best_fit = pp.bestfit
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_model", best_fit)

        red_chi2 = pp.chi2
        print("Rough reduced chi-squared from ppxf: " + pp.chi2)
       
        data_to_file = f.getvalue()

        kinematics_file.write(data_to_file)
        kinematics_file.write("")

        kinematics_file.write("Formal errors: \n")
        kinematics_file.write("     dV    dsigma   dh3      dh4 \n")
        kinematics_file.write("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)) 
                + "\n")

        kinematics_file.write('Elapsed time in PPXF: %.2f s' % (process_time() - t) 
                + "\n")

        graph_loc = "ppxf_results" + "/cube_" + str(int(cube_id))
        if not os.path.exists(graph_loc):
            os.mkdir(graph_loc) 

        kinematics_graph = (graph_loc + "/cube_" + str(int(cube_id)) + 
                "_kinematics.pdf")
        plt.savefig(kinematics_graph)
        #plt.show()
        plt.close("all") 


    # If the galaxy is at significant redshift z and the wavelength has been
    # de-redshifted with the three lines "z = 1.23..." near the beginning of
    # this procedure, the best-fitting redshift is now given by the following
    # commented line (equation 2 of Cappellari et al. 2009, ApJ, 704, L34):
    #
    #print, 'Best-fitting redshift z:', (z + 1)*(1 + sol[0]/c) - 1

    return {'reduced_chi2': red_chi2, 'noise': noise}

#------------------------------------------------------------------------------

#if __name__ == '__main__':
    #ppxf_example_kinematics_sdss(468)
    #import matplotlib.pyplot as plt
    #plt.pause(1)
    #plt.show()
