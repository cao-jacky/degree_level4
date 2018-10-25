import glob
from os import path
from time import clock

from astropy.io import fits
from scipy import ndimage
import numpy as np

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

def ppxf_example_kinematics_sauron():

    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

    # Read a galaxy spectrum and define the wavelength range
    #

    cube_id = 468
    cube_file = "data/cubes/cube_" + str(cube_id) + ".fits"
    hdu = fits.open(cube_file)
    gal_lin = hdu[0].data
    h1 = hdu[0].header

    cube_x_data = np.load("results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbd_x.npy")
    cube_y_data = np.load("results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbs_y.npy")

    mask = (cube_x_data > 5000) & (cube_x_data < 6000)
    cube_y_data = cube_y_data[mask]
 
    lamRange1 = h1['CRVAL1'] + np.array([0., np.abs(h1['CD1_1'])*(h1['NAXIS1'] - 1)])
    FWHM_gal = 2.51  # SAURON has an instrumental resolution FWHM of 4.2A.

    # If the galaxy is at significant redshift, one should bring the galaxy
    # spectrum roughly to the rest-frame wavelength, before calling pPXF
    # (See Sec2.4 of Cappellari 2017). In practice there is no
    # need to modify the spectrum in any way, given that a red shift
    # corresponds to a linear shift of the log-rebinned spectrum.
    # One just needs to compute the wavelength range in the rest-frame
    # and adjust the instrumental resolution of the galaxy observations.
    # This is done with the following three commented lines:
    #
    # z = 1.23 # Initial estimate of the galaxy redshift
    # lamRange1 = lamRange1/(1+z) # Compute approximate restframe wavelength range
    # FWHM_gal = FWHM_gal/(1+z)   # Adjust resolution in Angstrom

    galaxy, logLam1, velscale = util.log_rebin(lamRange1, cube_y_data)
    
    galaxy = galaxy/np.median(galaxy)  # Normalize spectrum to avoid numerical issues
    noise = np.full_like(galaxy, 0.0047)           # Assume constant noise per pixel here

    print(galaxy)
    # Read the list of filenames from the Single Stellar Population library
    # by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
    # of the library is included for this example with permission
    vazdekis = glob.glob(ppxf_dir + '/miles_models/Mun1.30Z*.fits')
    FWHM_tem = 2.51  # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
    velscale_ratio = 2  # adopts 2x higher spectral sampling for templates than for galaxy

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to a velocity scale 2x smaller than the SAURON galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = fits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])
    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)
    templates = np.empty((sspNew.size, len(vazdekis)))

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SAURON and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels

    for j, file in enumerate(vazdekis):
        hdu = fits.open(file)
        ssp = hdu[0].data
        ssp = ndimage.gaussian_filter1d(ssp, sigma)
        sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)
        templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below (see above).
    #
    c = 299792.458
    if velscale_ratio > 1:
        dv = (np.mean(logLam2[:velscale_ratio]) - logLam1[0])*c  # km/s
    else:
        dv = (logLam2[0] - logLam1[0])*c  # km/s

    z = 0.0015  # Initial redshift estimate of the galaxy
    goodPixels = util.determine_goodpixels(logLam1, lamRange2, z)

    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #
    vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
    t = clock()

    pp = ppxf(templates, galaxy, noise, velscale, start,
              goodpixels=goodPixels, plot=True, moments=4,
              degree=4, vsyst=dv, velscale_ratio=velscale_ratio)

    print("Formal errors:")
    print("     dV    dsigma   dh3      dh4")
    print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

    print('Elapsed time in pPXF: %.2f s' % (clock() - t))

    # If the galaxy is at significant redshift z and the wavelength has been
    # de-redshifted with the three lines "z = 1.23..." near the beginning of
    # this procedure, the best-fitting redshift is now given by the following
    # commented line (equation 2 of Cappellari et al. 2009, ApJ, 704, L34;
    # http://adsabs.harvard.edu/abs/2009ApJ...704L..34C)
    #
    # print('Best-fitting redshift z:', (z + 1)*(1 + pp.sol[0]/c) - 1)

#------------------------------------------------------------------------------

if __name__ == '__main__':

    ppxf_example_kinematics_sauron()
    import matplotlib.pyplot as plt
    #plt.pause(1)
    plt.show()
