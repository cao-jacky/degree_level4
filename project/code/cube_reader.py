import numpy as np
from numpy import unravel_index

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from astropy.io import fits

def read_file(file_name):
    # reads file_name and returns specific header data and image data

    fits_file = fits.open(file_name)

    header = fits_file[0].header
    image_data = fits_file[0].data

    header_keywords = {'CRVAL3': 0, 'CRPIX3': 0, 'CD3_3': 0}
    # clause to differentiate between CDELT3 and CD3_3

    for hdr_key, hdr_value in header_keywords.items():
        # finding required header values
        hdr_value = header[hdr_key]
        header_keywords[hdr_key] = hdr_value

    return header_keywords, image_data

def wavelength_solution(file_name):
    # wavelength solution in Angstroms

    file_data   = read_file(file_name)
    header_data = file_data[0]
    image_data  = file_data[1]

    range_begin = header_data['CRVAL3']
    pixel_begin = header_data['CRPIX3']
    step_size   = header_data['CD3_3']

    steps       = len(image_data)

    range_end   = range_begin + steps * step_size

    return {'begin': range_begin, 'end': range_end, 'steps': steps}

def image_collapser(file_name):

    file_data   = read_file(file_name)
    header_data = file_data[0]
    image_data  = file_data[1]

    data_shape  = np.shape(image_data)
    ra_axis     = data_shape[2]
    dec_axis    = data_shape[1]
    wl_axis     = data_shape[0]
    
    image_median    = np.zeros((ra_axis, dec_axis))
    image_sum       = np.zeros((ra_axis, dec_axis))

    for i_ra in range(ra_axis):
        for i_dec in range(dec_axis):
            pixel_data  = image_data[:][:,i_dec][:,i_ra]
            pd_median   = np.median(pixel_data)
            pd_sum      = np.sum(pixel_data)

            image_median[i_ra][i_dec]   = pd_median
            image_sum[i_ra][i_dec]      = pd_sum

    return {'median': image_median, 'sum': image_sum}

def spectrum_creator(file_name):
   
    file_data   = read_file(file_name)
    image_data  = file_data[1]

    collapsed_data  = image_collapser(file_name)

    # spectrum for central pixel
    cp_bright = []
    for key, data in collapsed_data.items():
        lgst_val = data.argmax()
        lgst_loc = unravel_index(data.argmax(), data.shape)
        cp_bright.append(lgst_loc)

    cp_loc = 0
    if ( cp_bright[0] == cp_bright[1] ):
        cp_loc = cp_bright[0]

    cp_spec_data    = image_data[:][:,cp_loc[0]][:,cp_loc[1]]
   
    # galaxy integrated spectrum
    gal_lim = [int(x / 2) for x in cp_loc]

    gal_cs_data   = image_data[:,gal_lim[0]:cp_loc[0]+gal_lim[0],
            gal_lim[1]:cp_loc[1]+gal_lim[1]]
    gs_shape = np.shape(gal_cs_data)

    gs_data = np.zeros(gs_shape[0])
    for i_ax in range(gs_shape[0]):
        col_data = gal_cs_data[i_ax][:]
        gs_data[i_ax] = np.sum(col_data)

    return {'central': cp_spec_data, 'galaxy': gs_data}

def spectra_stacker(file_name):
    
    file_data   = read_file(file_name)
    image_data  = file_data[1]

    data_shape  = np.shape(image_data)
    ra_axis     = data_shape[2]
    dec_axis    = data_shape[1]
    wl_axis     = data_shape[0]

    pxl_total   = ra_axis * dec_axis
    
    data_unwrap = []

    for i_ra in range(ra_axis):
        for i_dec in range(dec_axis):
            pixel_data  = image_data[:][:,i_dec][:,i_ra]
            
            data_unwrap.append(pixel_data)

    return data_unwrap

def sky_noise(sky_file_name):
    fits_file = fits.open(sky_file_name)
    image_data = fits_file[0].data

    return image_data

def spectra_analysis(file_name, sky_file_name):

    # spectra and sky noise data
    spectra_data = spectrum_creator(file_name)
    sn_data = sky_noise(sky_file_name)

    galaxy_data   = spectra_data['galaxy']

    # shifting the data down to be approximately on y=0 
    gd_mc   = np.average(galaxy_data) 
    gd_mc   = galaxy_data - gd_mc

    # scaling sky-noise to be similar to spectra data
    gd_max   = np.amax(galaxy_data)
    sn_data_max = np.amax(sn_data)
    sn_scale    = gd_max / sn_data_max

    sn_data     = sn_data * sn_scale

    return {'gd_shifted': gd_mc, 'sky_noise': sn_data}


def graphs(file_name, sky_file_name):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    # for collapsed images
    im_coll_data = image_collapser(file_name)

    smfig = plt.figure(1)
    plt.imshow(im_coll_data['median'], cmap='gray') 
    plt.title(r'\textbf{galaxy: median}', fontsize=13)    
    plt.xlabel(r'\textbf{Pixels}', fontsize=13)
    plt.ylabel(r'\textbf{Pixels}', fontsize=13)
    plt.savefig('graphs/collapse_median.pdf')

    ssfig = plt.figure(2)
    plt.imshow(im_coll_data['sum'], cmap='gray')
    plt.title(r'\textbf{galaxy: sum}', fontsize=13)        
    plt.xlabel(r'\textbf{Pixels}', fontsize=13)
    plt.ylabel(r'\textbf{Pixels}', fontsize=13)
    plt.savefig('graphs/collapse_sum.pdf')

    # spectra
    spectra_data = spectrum_creator(file_name)
    sr = wavelength_solution(file_name) #spectra_range
    
    cp_spec = plt.figure(3)
    cps_x   = np.linspace(sr['begin'], sr['end'], sr['steps'])
    cps_y   = spectra_data['central']
    plt.title(r'\textbf{spectra: central point}', fontsize=13)    
    plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
    #plt.ylabel(r'\textbf{}', fontsize=13)
    plt.plot(cps_x, cps_y, linewidth=0.5, color="#000000")
    plt.savefig('graphs/spectra_central_pixel.pdf')

    gs_data = spectra_analysis(file_name, sky_file_name)

    cp_spec = plt.figure(4)
    cps_x   = np.linspace(sr['begin'], sr['end'], sr['steps'])
     
    ## plotting our cube data
    cps_y   = gs_data['gd_shifted']
    plt.plot(cps_x, cps_y, linewidth=0.5, color="#000000")

    ## plotting our sky noise data
    sn_y    =  gs_data['sky_noise']
    plt.plot(cps_x, sn_y, linewidth=0.5, color="#e53935")

    plt.title(r'\textbf{spectra: cross-section}', fontsize=13)        
    plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
    plt.ylim(-500,5000) # setting manual limits for now
    plt.savefig('graphs/spectra_galaxy.pdf')

    # unwrapped 2d data
    unwrap_data = spectra_stacker(file_name)
    #reusing wavelength solution from above

    unwp    = plt.figure(5, figsize=(8, 38))
    for i in range(len(unwrap_data)): 
        unwp_x   = np.linspace(sr['begin'], sr['end'], sr['steps'])
        unwp_y   = unwrap_data[i] + i * 100
        plt.plot(unwp_x, unwp_y, linewidth=0.5, color=np.random.rand(3,))

    plt.title(r'\textbf{unwrapped 2d data}', fontsize=13)        
    plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
    plt.savefig('graphs/unwrap_2d.pdf')
     

graphs("data/cube_23.fits", "data/skyvariance_csub.fits")
