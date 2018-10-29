from time import process_time

import numpy as np

from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

def read_cube(file_name):
    fits_file = fits.open(file_name)

    header = fits_file[1].header
    data = fits_file[1].data
    return {'header': header, 'data': data}

def read_segmentation(file_name):
    fits_file = fits.open(file_name)

    header = fits_file[0].header
    data = fits_file[0].data
    return {'header': header, 'data': data}

def wavelength_solution(header, data_length):
    range_begin = header['CRVAL3']
    pixel_begin = header['CRPIX3']
    step_size   = header['CD3_3']

    steps       = data_length
    range_end   = range_begin + steps * step_size
    return {'begin': range_begin, 'end': range_end, 'steps': steps}

def cube_extractor(file_name):
    read_fits_file = read_cube(file_name)
    read_seg_file = read_segmentation("data/segmentation.fits")

    header = read_fits_file['header']
    data = read_fits_file['data']

    sheader = read_seg_file['header']
    sdata = read_seg_file['data']

    # rotating and flipping numpy array
    #sdata = np.flip(sdata, 0)
    #sdata = np.rot90(sdata, 1)

    #wls = wavelength_solution(header, len(data))

    #print(np.shape(data), np.shape(sdata))
    print(sdata)

    catalogue = np.load("data/matched_catalogue.npy")

    t = process_time()
    for i_obj in range(len(catalogue)):
        cube_id = int(catalogue[i_obj][0])
        print("Currently creating cube for " + str(cube_id))
        x_posn = int(np.rint(catalogue[i_obj][1]))
        y_posn = int(np.rint(catalogue[i_obj][2]))

        rl = [20,20] # region limits for x and y directions, makes a square

        subcube_data = data[:,y_posn-rl[0]:y_posn+rl[1],x_posn-rl[0]:x_posn+rl[1]]
        segmentation_data = sdata[y_posn-rl[0]:y_posn+rl[1],x_posn-rl[0]:x_posn+rl[1]]

        if (i_obj == 0):
            print(cube_id)
            print(x_posn, y_posn)
            print(y_posn-rl[0],y_posn+rl[1],x_posn-rl[0],x_posn+rl[1])
            np.set_printoptions(threshold=np.nan)
            print(segmentation_data)

        # saving both sets of data to one fits file
        # creating the header
        hdr = fits.Header()
        hdr['CTYPE3'] = 'AWAV'
        hdr['CRVAL3'] = header['CRVAL3']
        hdr['CRPIX3'] = header['CRPIX3']
        hdr['CD3_3'] = header['CD3_3']

        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdu1 = fits.ImageHDU(subcube_data)
        hdu2 = fits.ImageHDU(segmentation_data)

        hdul = fits.HDUList([primary_hdu, hdu1, hdu2])

        hdul.writeto("/Volumes/Jacky_Cao/University/level4/project/cubes_better/cube_"
                + str(cube_id) + ".fits" )

    print('Elapsed time in creating cubes: %.2f s' % (process_time() - t))


cube_extractor("/Volumes/Jacky_Cao/University/level4/project/DATACUBE_UDF-MOSAIC.fits")

    
