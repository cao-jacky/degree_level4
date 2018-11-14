import numpy as np

from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

def fits_noise_converter(sky_file_name):
    fits_file = fits.open(sky_file_name)

    header = fits_file[1].header
    data = fits_file[1].data
    data_shape = np.shape(data)

    ra_axis     = data_shape[2]
    dec_axis    = data_shape[1]
    wl_axis     = data_shape[0]

    pxl_total   = ra_axis * dec_axis

    # unwrapping data cube into a 2D array where each row is a pixel and columns 
    # represent pixels from the spectrograph
    data_unwrap = [] 
    for i_ra in range(ra_axis):
        for i_dec in range(dec_axis):
            pixel_data  = data[:][:,i_dec][:,i_ra]
            
            data_unwrap.append(pixel_data)

    # converting the data_unwrap list into an array
    data_stacked = np.zeros((pxl_total, wl_axis))
    for i_row in range(np.shape(data_unwrap)[0]):
        data_row = data_unwrap[i_row]
        for i_pixel in range(len(data_row)):
            data_stacked[i_row][i_pixel] = data_row[i_pixel]

    # finding the median from each spectrographic pixel and storing into an array
    noise_spectrum = np.zeros(wl_axis)
    for i in range(data_shape[0]):
        # I want to run through the z-axis (the spectroscopic data) and find the median
        # from each of the graphs
        #noise_spectrum[i] = np.median(data_stacked[:,i])
        noise_spectrum[i] = np.std(data_stacked[:,i])
    
    print(noise_spectrum)

    hdr = fits.Header()
    hdr['CTYPE1'] = 'pixel'
    hdr['CRPIX1'] = 1
    hdr['CRVAL1'] = data_stacked[0][0]
    hdr['CDELT1'] = data_stacked[0][1] - data_stacked[0][0]

    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdu1 = fits.ImageHDU(noise_spectrum)
    hdul = fits.HDUList([primary_hdu, hdu1])

    hdul.writeto("/Volumes/Jacky_Cao/University/level4/project/cube_noise_std.fits" ) 
    
fits_noise_converter("/Volumes/Jacky_Cao/University/level4/project/sky_noise.fits")

