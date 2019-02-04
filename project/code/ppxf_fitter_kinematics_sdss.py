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

import ntpath

import spectra_data

def cube_noise():
    cube_noise_file = "data/cube_noise_std.fits"
    cube_noise_file = fits.open(cube_noise_file)

    cube_noise_data = cube_noise_file[1].data 
    noise = np.sum(np.abs(cube_noise_data))
    return {'noise_value': noise, 'spectrum_noise': cube_noise_data}

def fitting_plotter(cube_id, ranges, x_data, y_data, x_model, y_model, noise):
    # parameters from lmfit
    lm_params = spectra_data.lmfit_data(cube_id)
    c = lm_params['c']
    i1 = lm_params['i1']
    i2 = lm_params['i2']
    sigma_gal = lm_params['sigma_gal']
    z = lm_params['z']
    sigma_inst = lm_params['sigma_inst']

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
    noise_median = np.median(noise)
    noise_stddev = np.std(noise) 

    residual = y_data_scaled - y_model
    print(x_data, x_model)
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

    plt.figure()

    plt.plot(x_data, y_data_scaled, linewidth=1.1, color="#000000")
    plt.plot(x_data, y_data_scaled+noise_stddev, lw=0.1, c="#616161", alpha=0.1)
    plt.plot(x_data, y_data_scaled-noise_stddev, lw=0.1, c="#616161", alpha=0.1)
    
    # plotting over the OII doublet
    doublets = np.array([3727.092, 3728.875])
    dblt_av = np.average(doublets) * (1+z)

    if (ranges[0] > dblt_av):
        pass
    else:
        dblt_x_mask = ((x_data > dblt_av-20) & (x_data < dblt_av+20))
        doublet_x_data = x_data[dblt_x_mask]
        doublet_data = spectra_data.f_doublet(doublet_x_data, c, i1, i2, sigma_gal, 
                z, sigma_inst)
        doublet_data = doublet_data / np.median(y_data)
        plt.plot(doublet_x_data, doublet_data, linewidth=0.5, color="#9c27b0")

    max_y = np.max(y_data_scaled) 

    plt.plot(x_model, y_model, linewidth=1.5, color="#b71c1c")

    residuals_mask = (residual > res_stddev) 
    rmask = residuals_mask

    #plt.scatter(x_data[rmask], residual[rmask], s=3, color="#f44336", alpha=0.5)
    plt.scatter(x_data[mask], residual[mask]-1, s=3, color="#43a047")

    plt.xlim([ranges[0], ranges[1]])

    plt.tick_params(labelsize=15)
    plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=15)
    plt.ylabel(r'\textbf{Relative Flux}', fontsize=15)
    plt.tight_layout()
 
    # range specifier for file name
    range_string = str(ranges[0]) + "_" + str(ranges[1])

    plt.savefig("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + str(int(cube_id))
            + "_" + range_string + "_fitted.pdf")

    plt.close("all")

##############################################################################

def kinematics_sdss(cube_id, y_data_var, fit_range):     
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
    lmfitd = spectra_data.lmfit_data(cube_id)
    z = lmfitd['z']

    cube_x_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbd_x.npy") 
    if (np.sum(y_data_var) == 0):
        cube_y_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbs_y.npy")
    else:
        cube_y_data = y_data_var

    #cube_x_data = cube_x_data / (1+z)

    cube_x_original = cube_x_data
    cube_y_original = cube_y_data

    # calculating the signal to noise
    sn_region = np.array([4000, 4080]) * (1+z) 
    sn_region_mask = ((cube_x_data > sn_region[0]) & (cube_x_data < sn_region[1]))
    
    cube_y_sn_region = cube_y_data[sn_region_mask]
    cy_sn_mean = np.mean(cube_y_sn_region)
    cy_sn_std = np.std(cube_y_sn_region)
    cy_sn = cy_sn_mean / cy_sn_std

    # cube noise
    cube_noise_data = cube_noise()
    spectrum_noise = cube_noise_data['spectrum_noise'] 
    spec_noise = spectrum_noise 

    lamRange = np.array([np.min(cube_x_data), np.max(cube_x_data)]) 
    specNew, logLam, velscale = log_rebin(lamRange, cube_y_data)
    lam = np.exp(logLam)
    
    loglam = np.log10(lam)
    # Only use the wavelength range in common between galaxy and stellar library.
    mask = (loglam > np.log10(3460)) & (loglam < np.log10(9464))
    flux = specNew[mask]
 
    galaxy = flux/np.median(flux)   # Normalize spectrum to avoid numerical issues
    loglam_gal = loglam[mask]
    lam_gal = 10**loglam_gal

    # galaxy spectrum not scaled 
    galaxy_ns = flux
    
    segmentation_data = hdu[2].data
    seg_loc_rows, seg_loc_cols = np.where(segmentation_data == cube_id)
    signal_pixels = len(seg_loc_rows) 

    spec_noise = spec_noise[mask]

    noise = np.nan_to_num((spec_noise * np.sqrt(signal_pixels)) / np.median(flux))

    # Considering specific ranges
    if (isinstance(fit_range, str)):
        pass
    else: 
        rtc_mask = ((loglam > np.log10(fit_range[0])) & 
                (loglam < np.log10(fit_range[1])))

        true_pixels = np.where(rtc_mask == True)[0]

        for i_noise in range(len(noise)):
            curr_noise = noise[i_noise]
            if i_noise in true_pixels:
                pass
            else:
                noise[i_noise] = 1.0

    # sky noise
    sky_noise = cube_reader.sky_noise("data/skyvariance_csub.fits")
    skyNew, skyLogLam, skyVelScale = log_rebin(lamRange, sky_noise)
    skyNew = skyNew[mask]

    c = 299792.458                  # speed of light in km/s
    frac = lam_gal[1]/lam_gal[0]    # Constant lambda fraction per pixel
    dlam_gal = (frac - 1)*lam_gal   # Size of every pixel in Angstrom

    data_shape = np.shape(galaxy)
    wdisp = np.full(data_shape, 1, dtype=float) # Intrinsic dispersion of every pixel

    sky_sigma_inst = np.load("data/sigma_inst.npy")
    fwhm_gal = 2.35*sky_sigma_inst*wdisp

    velscale = np.log(frac)*c       # Constant velocity scale in km/s per pixel

    # If the galaxy is at significant redshift, one should bring the galaxy
    # spectrum roughly to the rest-frame wavelength, before calling pPXF
    # (See Sec2.4 of Cappellari 2017). In practice there is no
    # need to modify the spectrum in any way, given that a red shift
    # corresponds to a linear shift of the log-rebinned spectrum.
    # One just needs to compute the wavelength range in the rest-frame
    # and adjust the instrumental resolution of the galaxy observations.
    # This is done with the following three commented lines:
    
    lam_gal = lam_gal/(1+z)  # Compute approximate restframe wavelength
    fwhm_gal = fwhm_gal/(1+z)   # Adjust resolution in Angstrom

    # Read the list of filenames from the Single Stellar Population library
    # by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
    # of the library is included for this example with permission

    # NOAO Coudé templates
    template_set = glob.glob("noao_templates/*.fits")
    fwhm_tem = 1.35 

    # Extended MILES templates
    #template_set = glob.glob('miles_models/s*.fits') 
    #fwhm_tem = 2.5
    
    # Jacoby templates
    #template_set = glob.glob('jacoby_models/jhc0*.fits')
    #fwhm_tem = 4.5 # instrumental resolution in Ångstroms.

    # Default templates
    #template_set = glob.glob('miles_models/Mun1.30Z*.fits')
    #fwhm_tem = 2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the SDSS galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = fits.open(template_set[0])
    
    noao_data = hdu[1].data[0]
    ssp = noao_data[1]

    lam_temp = noao_data[0]

    lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
    sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
    templates = np.empty((sspNew.size, len(template_set)))

    # Interpolates the galaxy spectral resolution at the location of every pixel
    # of the templates. Outside the range of the galaxy spectrum the resolution
    # will be extrapolated, but this is irrelevant as those pixels cannot be
    # used in the fit anyway.
    fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)

    # Convolve the whole library of spectral templates
    # with the quadratic difference between the data and the
    # template instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels 
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    # In the line below, the fwhm_dif is set to zero when fwhm_gal < fwhm_tem.
    # In principle it should never happen and a higher resolution template should be used
    #
    fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0)) 
    
    spacing = lam_temp[1] - lam_temp[0]
    sigma = fwhm_dif/2.355/spacing # Sigma difference in pixels
    for j, fname in enumerate(template_set):
        hdu = fits.open(fname)
        #ssp = hdu[0].data

        noao_data = hdu[1].data[0]
        ssp = noao_data[1]

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
    dv = np.log(lam_temp[0]/(lam_gal[0]*(1+z)))*c    # km/s
    goodpixels = util.determine_goodpixels(np.log(lam_gal*(1+z)), lamRange_temp, z) 

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
            degree=12, vsyst=dv, clean=True, lam=lam_gal) 

    ppxf_variables = pp.sol
    ppxf_errors = pp.error
 
    red_chi2 = pp.chi2
    best_fit = pp.bestfit

    x_data = cube_x_data[mask]
    y_data = cube_y_data[mask]

    print(ppxf_variables)
    #plt.show()
    
    if ((np.sum(y_data_var) == 0) and isinstance(fit_range, str)):
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_ppxf_variables", 
                ppxf_variables) 
 
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_galaxy", 
                galaxy)

        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_lamgal", lam_gal) 
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_flux", flux)
 
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_x", x_data)
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_y", y_data)

        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_noise", noise)
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_spec_noise", spec_noise)

        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_not_scaled", galaxy_ns)

        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_goodpixels", goodpixels)

        # if best fit i.e. perturbation is 0, save everything
     
        kinematics_file = open(file_loc + "/cube_" + str(int(cube_id)) + 
            "_kinematics.txt", 'w')

        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_model", best_fit)

        print("Rough reduced chi-squared from ppxf: " + str(pp.chi2))
       
        data_to_file = f.getvalue()

        kinematics_file.write(data_to_file)
        kinematics_file.write("")

        kinematics_file.write("Formal errors: \n")
        kinematics_file.write("     dV    dsigma   dh3      dh4 \n")
        kinematics_file.write("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)) 
                + "\n")

        kinematics_file.write('Elapsed time in PPXF: %.2f s' % (process_time() - t) 
                + "\n")

        plt.tight_layout()
        graph_loc = "ppxf_results" + "/cube_" + str(int(cube_id))
        if not os.path.exists(graph_loc):
            os.mkdir(graph_loc) 

        kinematics_graph = (graph_loc + "/cube_" + str(int(cube_id)) + 
                "_kinematics.pdf")
        plt.savefig(kinematics_graph)
        #plt.show()
        plt.close("all")
    if not isinstance(fit_range, str):
        # saving graphs if not original range
        fit_range = fit_range
        fitting_plotter(cube_id, fit_range, x_data, y_data, lam_gal, best_fit, noise)
        
        # goodpixels range specifier for array
        np.save(file_loc + "/cube_" + str(int(cube_id)) + "_goodpixels_" + 
                str(fit_range[0]) + "_" + str(fit_range[1]), goodpixels)

    # save the goodpixels array depending on if a reduced ranged is being used

    # If the galaxy is at significant redshift z and the wavelength has been
    # de-redshifted with the three lines "z = 1.23..." near the beginning of
    # this procedure, the best-fitting redshift is now given by the following
    # commented line (equation 2 of Cappellari et al. 2009, ApJ, 704, L34):
    #
    print
    print('Best-fitting redshift z: '+str((z + 1)*(1 + ppxf_variables[0]/c) - 1))

    return {'reduced_chi2': red_chi2, 'noise': noise, 'variables': ppxf_variables,
            'y_data': galaxy, 'x_data': lam_gal, 'redshift': z, 
            'y_data_original': cube_y_original, 'non_scaled_y': galaxy_ns,
            'model_data': best_fit, 'noise_original': spec_noise,
            'errors': ppxf_errors}

#------------------------------------------------------------------------------

#kinematics_sdss(1804, 0, "all")

#if __name__ == '__main__':
    #ppxf_example_kinematics_sdss(468)
    #import matplotlib.pyplot as plt
    #plt.pause(1)
    #plt.show()
