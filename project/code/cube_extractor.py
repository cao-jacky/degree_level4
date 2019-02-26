from time import process_time

import numpy as np

from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

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
    # sorting catalogue by the probability that the object is a star
    catalogue = catalogue[catalogue[:,8].argsort()]

    t = process_time()
    for i_obj in range(len(catalogue)):
        # we just want to consider the first 300 objects
        if (i_obj <= 300):
            cube_id = int(catalogue[i_obj][0])
            print("Currently creating cube for " + str(cube_id))
            x_posn = int(np.rint(catalogue[i_obj][1]))
            y_posn = int(np.rint(catalogue[i_obj][2]))

            rl = [25,25] # region limits for x and y directions, makes a square

            subcube_data = data[:,y_posn-rl[0]:y_posn+rl[1],x_posn-rl[0]:x_posn+rl[1]]
            segmentation_data = sdata[y_posn-rl[0]:y_posn+rl[1],
                    x_posn-rl[0]:x_posn+rl[1]]

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

            hdul.writeto("/Volumes/Jacky_Cao/University/level4/project/cubes_better/" 
                    + "cube_" + str(cube_id) + ".fits" )

    print('Elapsed time in creating cubes: %.2f s' % (process_time() - t))

def colour_image_data_extractor(file_name):
    read_fits_file = read_cube(file_name)
    data = read_fits_file['data']
    
    data_shape = np.shape(data)
    spectro_length = data_shape[0]
    spectro_third = int(spectro_length / 3)

    r = (data[0:spectro_third, :, :])
    np.save("/Volumes/Jacky_Cao/University/level4/project/master_cube_r", r)

    g = (data[spectro_third:spectro_third*2, :, :])
    np.save("/Volumes/Jacky_Cao/University/level4/project/master_cube_g", g)

    b = (data[spectro_third*2:spectro_third*3, :, :])
    np.save("/Volumes/Jacky_Cao/University/level4/project/master_cube_b", b)

def colour_image_collapser():
    r = np.load("/Volumes/Jacky_Cao/University/level4/project/master_cube_r.npy")
    r_new = np.zeros((np.shape(r)[1], np.shape(r)[2]))

    for i_r_x in range(np.shape(r)[1]):
        for i_r_y in range(np.shape(r)[2]):
            r_new[i_r_x][i_r_y] = np.nansum(r[:][:,i_r_x][:,i_r_y])
    np.save("data/frame_r_sum", r_new)

    g = np.load("/Volumes/Jacky_Cao/University/level4/project/master_cube_g.npy")
    g_new = np.zeros((np.shape(g)[1], np.shape(g)[2]))

    for i_g_x in range(np.shape(g)[1]):
        for i_g_y in range(np.shape(g)[2]):
            g_new[i_g_x][i_g_y] = np.nansum(g[:][:,i_g_x][:,i_g_y])
    np.save("data/frame_g_sum", g_new)

    b = np.load("/Volumes/Jacky_Cao/University/level4/project/master_cube_b.npy")
    b_new = np.zeros((np.shape(b)[1], np.shape(b)[2]))

    for i_b_x in range(np.shape(b)[1]):
        for i_b_y in range(np.shape(b)[2]):
            b_new[i_b_x][i_b_y] = np.nansum(b[:][:,i_b_x][:,i_b_y])
    np.save("data/frame_b_sum", b_new)

def colour_image():
    r = np.load("data/frame_r.npy")
    g = np.load("data/frame_g.npy")
    b = np.load("data/frame_b.npy")

    rgb_array = np.zeros((r.shape[0], r.shape[1], 3), dtype=float)

    fig = plt.figure()
    fig.set_size_inches(10,10)
    rgb_array[:,:,0] = r 
    rgb_array[:,:,1] = g 
    rgb_array[:,:,2] = b 
    plt.imshow(rgb_array, interpolation='nearest', origin='lower')
    plt.axis('off')
    plt.savefig('results/cube_colour_imshow.pdf', dpi=(500), bbox_inches='tight',
            pad_inches=0.0)

def noise_cube_extractor(file_name):
    read_fits_file = read_cube(file_name)
    read_seg_file = read_segmentation("data/segmentation.fits")

    header = read_fits_file['header']
    data = read_fits_file['data']

    sheader = read_seg_file['header']
    sdata = read_seg_file['data']

    t = process_time()

    top_left = [865, 328]
    bottom_right = [905, 294]

    print(bottom_right[1],top_left[1],top_left[0],bottom_right[0])

    subcube_data = data[:,bottom_right[1]:top_left[1],top_left[0]:bottom_right[0]]
    segmentation_data = sdata[bottom_right[1]:top_left[1],top_left[0]:bottom_right[0]]

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

    hdul.writeto("/Volumes/Jacky_Cao/University/level4/project/sky_noise.fits" )

    print('Elapsed time in creating cubes: %.2f s' % (process_time() - t))

def hst_hudf_extractor(file_name):
    fits_file = fits.open(file_name)

    header = fits_file[0].header
    data = fits_file[0].data

    ds = np.shape(data) # data shape

    # data grid array to store various things
    # [0] : grid with pixel references
    # [1] : grid with RA references
    # [2] : grid with dec references
    data_grid = np.zeros([3, ds[0], ds[1]])
    print(np.shape(data_grid))

    # reference pixels
    rp_x = header['CRPIX1'] # x-coordinate of reference pixel
    rp_y = header['CRPIX2'] # y-coordinate of reference pixel

    rp = data[int(rp_x-0.5):int(rp_x+0.5),int(rp_y-0.5):int(rp_y+0.5)]

    # x-axis
    xp_b = rp_x # x-coordinate of reference pixel 
    xv_b = header['CRVAL1'] # x-axis true value of reference pixel 
    xp_s = header['CD1_1'] # x-axis step size
    
    xa_s = ds[0] # x-axis number of steps

    # y-axis
    yp_b = rp_y # y-coordinate of reference pixel 
    yv_b = header['CRVAL2'] # y-axis true value of reference pixel 
    yp_s = header['CD2_2'] # y-axis step size
    
    ya_s = ds[1] # y-axis number of steps


    # creating x-axis in units of RA
    #xaxis = np.linspace(, wl_soln['end'], wl_soln['steps'])

    #steps       = data_length
    #range_end   = range_begin + steps * step_size

    print(header)
    print(rp)
    print(xv_b, yv_b)

    catalogue = np.load("data/matched_catalogue.npy")
    #print(catalogue)

if __name__ == '__main__':
    #cube_extractor("/Volumes/Jacky_Cao/University/level4/project/DATACUBE_UDF-MOSAIC.fits")
    #colour_image_data_extractor("/Volumes/Jacky_Cao/University/level4/project/DATACUBE_UDF-MOSAIC.fits")

    #colour_image_collapser()
    #colour_image()

    #noise_cube_extractor("/Volumes/Jacky_Cao/University/level4/project/DATACUBE_UDF-MOSAIC.fits")

    hst_hudf_extractor("/Volumes/Jacky_Cao/University/level4/project/HST_HUDF/hlsp_xdf_hst_acswfc-30mas_hudf_f606w_v1_sci.fits")

