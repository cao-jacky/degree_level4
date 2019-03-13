import os
from time import process_time

import numpy as np

from scipy import ndimage

from astropy.io import fits
from astropy import wcs

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import matplotlib.pyplot as plt
from matplotlib import rc

import hst_udf

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

    #catalogue = np.load("data/matched_catalogue.npy")
    catalogue = np.load("data/low_redshift_catalogue.npy") # low-redshift catalogue
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

            """
            if (i_obj == 0):
                print(cube_id)
                print(x_posn, y_posn)
                print(y_posn-rl[0],y_posn+rl[1],x_posn-rl[0],x_posn+rl[1])
                np.set_printoptions(threshold=np.nan)
                print(segmentation_data)
            """

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

def colour_image_extractor():
    # HST frames
    frame_i = ("/Volumes/Jacky_Cao/University/level4/project/HST_HUDF/hlsp_xdf_hst_acswfc-30mas_hudf_f435w_v1_sci.fits")
    frame_v = ("/Volumes/Jacky_Cao/University/level4/project/HST_HUDF/hlsp_xdf_hst_acswfc-30mas_hudf_f606w_v1_sci.fits")
    frame_b = ("/Volumes/Jacky_Cao/University/level4/project/HST_HUDF/hlsp_xdf_hst_acswfc-30mas_hudf_f775w_v1_sci.fits")
    frames = {0: frame_i, 1: frame_v, 2: frame_b}

    # loading one of the MUSE colour frames to find out the size of it
    r = np.load("data/frame_r.npy")
    shape_muse = np.shape(r)

    catalogue = np.load("data/matched_catalogue.npy")
    # sorting catalogue by the probability that the object is a star
    catalogue = catalogue[catalogue[:,8].argsort()]

    for i_obj in range(len(catalogue)):
        # we just want to consider the first 300 objects
        if (i_obj <= 300):
            cube_id = int(catalogue[i_obj][0])

            print("Currently cube_"+str(cube_id))

            cube_dir = "cube_results/cube_"+str(cube_id)
            if not os.path.exists(cube_dir):
                # Only create colour images for cubes already processed
                pass
            else:
                rl = [250,250] # cutout limits 
                colour_data = np.zeros([3, rl[0]+rl[1], rl[0]+rl[1]])

                for i in range(3):
                    curr_frame = frames[i] # load current frame
                    fits_file = fits.open(curr_frame) # open frame

                    # read header and data
                    header = fits_file[0].header 
                    data = fits_file[0].data

                    shape_hst = np.shape(data)

                    scale_x = int(shape_hst[1]/shape_muse[1])
                    scale_y = int(shape_hst[0]/shape_muse[0])

                    # Parse the WCS keywords in the primary HDU
                    w = wcs.WCS(header)

                    posn_ra = catalogue[i_obj][16]
                    posn_dec = catalogue[i_obj][17]
                    wcs_coord = np.array([[posn_ra, posn_dec]])

                    # converted from RA-dec to integer pixel location
                    pixel_coord = w.wcs_world2pix(wcs_coord, 1).astype(int) 
                    posn_x = pixel_coord[0][0]
                    posn_y = pixel_coord[0][1]

                    subcube_data = data[posn_y-rl[0]:posn_y+rl[1],
                            posn_x-rl[0]:posn_x+rl[1]]
                    colour_data[i] = np.flipud(subcube_data)
                
                # rotating data so that it's approx same angle as MUSE data
                colour_data = ndimage.rotate(colour_data, angle=225, axes=(1,2), 
                            mode='nearest', reshape=False)
                # flipping the data
                #colour_data = np.flipud(colour_data)

                b = colour_data[2]
                v = colour_data[1]
                i = colour_data[0]
                coloured_image = hst_udf.mkcol(b,v,i, 0.99, 0.9)

                # save as array
                np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                        "_coloured_image_data.npy", coloured_image)
                
                # save as colour image
                plt.figure()
                plt.imshow(coloured_image, interpolation='nearest')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                        "_coloured_image.pdf")

                plt.close("all")

if __name__ == '__main__':
    cube_extractor("/Volumes/Jacky_Cao/University/level4/project/DATACUBE_UDF-MOSAIC.fits")
    #colour_image_data_extractor("/Volumes/Jacky_Cao/University/level4/project/DATACUBE_UDF-MOSAIC.fits")

    #colour_image_collapser()
    #colour_image()

    #noise_cube_extractor("/Volumes/Jacky_Cao/University/level4/project/DATACUBE_UDF-MOSAIC.fits")

    #hst_hudf_extractor("/Volumes/Jacky_Cao/University/level4/project/HST_HUDF/hlsp_xdf_hst_acswfc-30mas_hudf_f606w_v1_sci.fits")

    #colour_image_extractor()

