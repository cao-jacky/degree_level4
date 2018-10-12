import numpy as np
from numpy import unravel_index

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from scipy import signal
from scipy.optimize import leastsq

from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

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
    # collapses image data so it can be passed as a heatmap
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
    # creates a combined single spectra from an area around the 'central pixel' 
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
    # stacking all spectra together for a stacked spectra image
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
    # returning sky noise data files
    fits_file = fits.open(sky_file_name)
    image_data = fits_file[0].data

    return image_data

def spectra_analysis(file_name, sky_file_name):
    # correcting data to be in rest frame
    # spectra and sky noise data
    spectra_data    = spectrum_creator(file_name)
    wl_soln         = wavelength_solution(file_name)
    sn_data         = sky_noise(sky_file_name)

    galaxy_data   = spectra_data['galaxy']

    # shifting the data down to be approximately on y=0 
    gd_mc   = np.average(galaxy_data) 
    gd_mc   = galaxy_data - gd_mc

    # scaling sky-noise to be similar to spectra data
    gd_max   = np.amax(galaxy_data)
    sn_data_max = np.amax(sn_data)
    sn_scale    = gd_max / sn_data_max

    sn_data     = sn_data * sn_scale

    # spectra lines
    sl = {
            'emis': {
                '[OII]':      '3727',
                'CaK':      '3933',
                'CaH':      '3968',
                'Hdelta':   '4101',

                
                }, 
            'abs': {'K': '3934.777',
                }
            } 

    gd_peaks = signal.find_peaks_cwt(gd_mc, np.arange(10,15), noise_perc=100)
    #print("Peaks from galaxy data: ")
    #print(gd_peaks)

    # manually selecting which peak is the [OII] peak - given in wavelength
    otwo_wav    = float(wl_soln['begin'] + gd_peaks[7])    
    otwo_acc    = float(sl['emis']['[OII]'])

    redshift = (otwo_wav - otwo_acc) / otwo_acc

    return {'gd_shifted': gd_mc, 'sky_noise': sn_data, 'spectra': sl, 'gd_peaks': 
            gd_peaks, 'redshift': redshift}

def norm(x, mean, sd):
    norm = []
    for i in range(x.size):
        norm += [1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x[i] - mean)**2/(2*sd**2))]
    return np.array(norm)

def otwo_doublet_fitting(file_name, sky_file_name):
    sa_data     = spectra_analysis(file_name, sky_file_name)
    y_shifted   = sa_data['gd_shifted'] 
    orr         = wavelength_solution(file_name) 

    # OII range and region
    otr         = [1100, 1200] 
    otwo_region = y_shifted[otr[0]:otr[1]]

    # I need to have the x-data here as well, so copy the code from plotters
    orr_x   = np.linspace(orr['begin'], orr['end'], orr['steps'])
    rdst    = sa_data['redshift']
    corr_x  = orr_x / (1+rdst) # corrected wavelengths

    ot_x    = corr_x[otr[0]:otr[1]]

    # m, dm, std1, std2 
    init_test   = [ 3720, 3730, 10, 10 ]

    def res(p, y, x):
        m1, m2, sd1, sd2 = p
        y_fit = norm(x, m1, sd1) + norm(x, m2, sd2)
        y_fit = norm(x, m1, sd1) 
        err = y - y_fit
        return err

    plsq = leastsq(res, init_test, args=(otwo_region, ot_x))

    y_est = norm(ot_x, plsq[0][0], plsq[0][2]) + norm(ot_x, plsq[0][0] + plsq[0][1], 
            plsq[0][3])

    return {'range': otr, 'region': otwo_region, 'gauss_estimate': y_est }

def graphs(file_name, sky_file_name):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    # --- for collapsed images ---
    def graphs_collapsed():
        im_coll_data = image_collapser(file_name)

        smfig = plt.figure(1)
        plt.imshow(im_coll_data['median'], cmap='gray_r') 
        plt.title(r'\textbf{galaxy: median}', fontsize=13)    
        plt.xlabel(r'\textbf{Pixels}', fontsize=13)
        plt.ylabel(r'\textbf{Pixels}', fontsize=13)
        plt.savefig('graphs/collapse_median.pdf')

        ssfig = plt.figure(2)
        plt.imshow(im_coll_data['sum'], cmap='gray_r')
        plt.title(r'\textbf{galaxy: sum}', fontsize=13)        
        plt.xlabel(r'\textbf{Pixels}', fontsize=13)
        plt.ylabel(r'\textbf{Pixels}', fontsize=13)
        plt.savefig('graphs/collapse_sum.pdf')

    # --- spectra ---
    def graphs_spectra():
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

        # --- uncorrected redshift

        gs_data = spectra_analysis(file_name, sky_file_name)

        cp_spec = plt.figure(4)
        cps_x   = np.linspace(sr['begin'], sr['end'], sr['steps'])
         
        ## plotting our cube data
        cps_y   = gs_data['gd_shifted']
        plt.plot(cps_x, cps_y, linewidth=0.5, color="#000000")

        ## plotting our sky noise data
        sn_y    =  gs_data['sky_noise']
        plt.plot(cps_x, sn_y, linewidth=0.5, color="#e53935")

        ## plotting peak lines
        pk_lines = gs_data['gd_peaks']
        for i in range(len(pk_lines)):
            srb = sr['begin']
            plt.axvline(x=(srb+pk_lines[i]), linewidth=0.5, color="#8bc34a")

        ## shifting data
        
        plt.title(r'\textbf{spectra: cross-section redshifted}', fontsize=13)        
        plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
        plt.ylim(-500,5000) # setting manual limits for now
        plt.savefig('graphs/spectra_galaxy_redshifted.pdf')
    
        # --- corrected redshift
        df_data = otwo_doublet_fitting(file_name, sky_file_name) # sliced region

        cp_spec = plt.figure(5)
        cps_x   = np.linspace(sr['begin'], sr['end'], sr['steps'])
        rdst    = gs_data['redshift']

        sp_lines = gs_data['spectra']

        ## corrected wavelengths
        corr_x  = cps_x / (1+rdst)

        ## plotting our cube data
        cps_y   = gs_data['gd_shifted']
        #plt.plot(cps_x, cps_y, linewidth=0.5, color="#000000", alpha=0.1)
        plt.plot(corr_x, cps_y, linewidth=0.5, color="#000000")

        ## plotting our sky noise data
        sn_y    =  gs_data['sky_noise']
        #plt.plot(cps_x, sn_y, linewidth=0.5, color="#e53935", alpha=0.1)
        plt.plot(corr_x, sn_y, linewidth=0.5, color="#e53935")

        ## plotting our [OII] region
        ot_y    = df_data['region']
 
        ## corrected wavelengths
        corr_x  = cps_x / (1+rdst)

        dt_rng  = df_data['range']
        ot_xb   = corr_x[0] + dt_rng[0] # beginning of range
        ot_xe   = corr_x[0] + dt_rng[1] # end of range 

        ot_x    = corr_x[dt_rng[0]:dt_rng[1]]

        plt.plot(ot_x, ot_y, linewidth=0.5, color="#00c853")

        ## plotting spectra lines
        for e_key, e_val in sp_lines['emis'].items():
            spec_line = float(e_val)
            plt.axvline(x=spec_line, linewidth=0.5, color="#00c853")
            plt.text(spec_line-10, 4800, e_key, rotation=-90)
 
        plt.title(r'\textbf{spectra: cross-section corrected}', fontsize=13)        
        plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
        plt.ylim(-500,5000) # setting manual limits for now
        plt.savefig('graphs/spectra_galaxy_corrected.pdf')

    # --- unwrapped 2d data ---
    def graphs_unwrapped():
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

    def graphs_otwo_region():
        df_data = otwo_doublet_fitting(file_name, sky_file_name) # sliced region
        orr     = wavelength_solution(file_name) # original range
        gs_data = spectra_analysis(file_name, sky_file_name)

        ot_y    = df_data['region']

        orr_x, orr_step = np.linspace(orr['begin'], orr['end'], orr['steps'], 
                retstep=True)
        rdst    = gs_data['redshift']

        ## corrected wavelengths
        corr_x  = orr_x / (1+rdst)

        dt_rng  = df_data['range']
        ot_xb   = corr_x[0] + dt_rng[0] # beginning of range
        ot_xe   = corr_x[0] + dt_rng[1] # end of range 

        ot_x    = corr_x[dt_rng[0]:dt_rng[1]]

        est_data = df_data['gauss_estimate']

        ot_fig  = plt.figure(6)
        plt.plot(ot_x, ot_y, linewidth=0.5, color='#000000')
        plt.plot(ot_x, est_data, linewidth=0.5, color='#2196f3')
        plt.title(r'\textbf{[OII] region}', fontsize=13)        
        plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
        plt.ylim(-500,5000) # setting manual limits for now
        plt.savefig('graphs/otwo_region.pdf')

        

    #graphs_collapsed()
    graphs_spectra()
    #graphs_unwrapped()
    graphs_otwo_region()
     

graphs("data/cube_23.fits", "data/skyvariance_csub.fits")
#otwo_doublet_fitting("data/cube_23.fits", "data/skyvariance_csub.fits")
