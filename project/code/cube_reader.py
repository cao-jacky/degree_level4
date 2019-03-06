import os
import re

import file_writer

import numpy as np
from numpy import unravel_index

import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from scipy import signal
import peakutils

from lmfit import minimize, Parameters, Model

from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

def read_file(file_name):
    """ reads file_name and returns specific header data and image data """
    fits_file = fits.open(file_name)

    header = fits_file[0].header
    image_data = fits_file[1].data

    segmentation_data = fits_file[2].data

    header_keywords = {'CRVAL3': 0, 'CRPIX3': 0, 'CD3_3': 0}
    # clause to differentiate between CDELT3 and CD3_3

    for hdr_key, hdr_value in header_keywords.items():
        # finding required header values
        hdr_value = header[hdr_key]
        header_keywords[hdr_key] = hdr_value

    return header_keywords, image_data, segmentation_data

def wavelength_solution(file_name):
    """ wavelength solution in Angstroms """
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
    """ collapses image data so it can be passed as a heatmap """
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
            pd_median   = np.nanmedian(pixel_data)
            pd_sum      = np.nansum(pixel_data)

            image_median[i_ra][i_dec]   = pd_median
            image_sum[i_ra][i_dec]      = pd_sum

    return {'median': image_median, 'sum': image_sum}

def spectrum_creator(file_name):
    """ creating a spectra from the area as defined in the segementation area """
    file_data   = read_file(file_name)
    image_data  = file_data[1]

    segmentation_data = file_data[2]

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
    else: 
        cp_loc = cp_bright[1]

    cp_spec_data    = image_data[:][:,cp_loc[0]][:,cp_loc[1]]

    # spectrum as defined by the segmentation area
    curr_file_name = file_name.split('.')
    curr_file_name = curr_file_name[0].split('/')
    stk_f_n = curr_file_name[-1]
    cube_id = [int(x) for x in re.findall('\d+', stk_f_n)][0]

    # locating where the galaxy pixels are from the cube_id
    seg_curr_cube = np.where(segmentation_data == cube_id)
    scc_rows, scc_cols = seg_curr_cube

    #np.set_printoptions(threshold=np.nan)
    #print(segmentation_data)

    collapsed_spectrum = np.zeros([np.shape(image_data)[0], len(scc_rows)])
    for i_r in range(len(scc_rows)):
        # I want to pull out each pixel and store it into the collapsed spectrum array
        collapsed_spectrum[:,i_r] = image_data[:,scc_rows[i_r],scc_cols[i_r]]
    
    galaxy_spectrum = np.zeros(np.shape(image_data)[0])
    for i_ax in range(len(galaxy_spectrum)):
        galaxy_spectrum[i_ax] = np.nansum(collapsed_spectrum[i_ax])
 
    return {'central': cp_spec_data, 'galaxy': galaxy_spectrum, 
            'segmentation': segmentation_data}

def cube_noise(cube_id):
    cube_file_name = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/" + 
            "cube_" + str(cube_id) + ".fits")
    cube_file = read_file(cube_file_name)

    image_data = cube_file[1]
    collapsed_data = spectrum_creator(cube_file_name)

    segmentation_data = cube_file[2]

    pixels_data = np.where(segmentation_data == cube_id)
    pixels_noise = np.where(segmentation_data == 0)
    pn_rows, pn_cols = pixels_noise

    pd_num = np.shape(pixels_data)[1]  # number of pixels so that a ratio can be 
    pn_num = np.shape(pixels_noise)[1] # calculated
    
    # calculating the noise based off the segmentation data
    noise_spectra = np.zeros([np.shape(image_data)[0], len(pn_rows)])
    for i_noise in range(len(pn_rows)):
        noise_spectra[:,i_noise] = image_data[:,pn_rows[i_noise],pn_cols[i_noise]]

    nr_noise = np.zeros(np.shape(noise_spectra)[0])
    for i_ax in range(np.shape(noise_spectra)[0]):
        nr_noise[i_ax] = np.nansum(noise_spectra[i_ax])
        
    noise = np.median(nr_noise) * np.sqrt(pd_num**2/pn_num**2)
    return {'noise_data': nr_noise, 'noise_value': noise, 'pd_num': pd_num, 
            'pn_num': pn_num}

def spectra_stacker(file_name): 
    """ stacking all spectra together for a stacked spectra image """
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

    data_stacked = np.zeros((pxl_total, wl_axis))
    for i_row in range(np.shape(data_unwrap)[0]):
        data_row = data_unwrap[i_row]
        for i_pixel in range(len(data_row)):
            data_stacked[i_row][i_pixel] = data_row[i_pixel]

    # writing data to a fits file
    hdr = fits.Header()
    hdr['CTYPE1'] = 'pixel'
    hdr['CRPIX1'] = 1
    hdr['CRVAL1'] = data_stacked[0][0]
    hdr['CDELT1'] = data_stacked[0][1] - data_stacked[0][0]

    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdu = fits.ImageHDU(data_stacked)

    hdul = fits.HDUList([primary_hdu, hdu])

    curr_file_name = file_name.split('.')
    curr_file_name = curr_file_name[0].split('/')
    stk_f_n = curr_file_name[-1]
   
    data_dir = 'cube_results/' + stk_f_n
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        hdul.writeto(data_dir + '/stacked.fits')
    return data_unwrap

def sky_noise(sky_file_name):
    """ returning sky noise data files """
    fits_file = fits.open(sky_file_name)
    image_data = fits_file[0].data
    return image_data

def spectra_analysis(file_name, sky_file_name):
    """ correcting data to be in rest frame """  

    # read file name and select out the id that we are dealing with
    curr_file_name = file_name.split('.')
    curr_file_name = curr_file_name[0].split('/')
    stk_f_n = curr_file_name[-1]
    cube_id = int(re.search(r'\d+', stk_f_n).group())

    # read catalogue and obtain the HST redshift estimate
    catalogue = np.load("data/matched_catalogue.npy")
    cat_loc = np.where(catalogue[:,0] == cube_id)[0]
    cube_info = catalogue[cat_loc][0]
 
    hst_redshift = cube_info[7]

    # spectra and sky noise data
    spectra_data    = spectrum_creator(file_name)
    wl_soln         = wavelength_solution(file_name)
    sn_data         = sky_noise(sky_file_name)

    galaxy_data   = spectra_data['galaxy']

    # removing baseline from data
    base = peakutils.baseline(galaxy_data, 3)
    gd_mc = galaxy_data - base

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

    # we can use the redshift from the HST catalogue to define the region to search for
    # the doublet in

    # lower and upper bound on wavelength range
    lower_lambda = (1+hst_redshift)*3600
    upper_lambda = (1+hst_redshift)*3850

    # x-axis data
    data_h_range = np.linspace(wl_soln['begin'], wl_soln['end'], wl_soln['steps'])
    mask = (lower_lambda < data_h_range) & (data_h_range < upper_lambda) 

    lambda_data = data_h_range[mask]
    flux_data = gd_mc[mask] 
    
    # Finding peaks with PeakUtils
    pu_peaks = peakutils.indexes(flux_data, thres=600, thres_abs=True)
    pu_peaks_x = peakutils.interpolate(lambda_data, flux_data, pu_peaks)

    pu_peaks_x = np.sort(pu_peaks_x)
    pu_peaks_x = pu_peaks_x[lower_lambda < pu_peaks_x]
    pu_peaks_x = pu_peaks_x[pu_peaks_x < upper_lambda]
   
    data_dir = 'cube_results/' + stk_f_n
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    peaks_file = open(data_dir + '/' + stk_f_n + '_peaks.txt', 'w')
    peaks_file.write("Peaks found on " + str(datetime.datetime.now()) + "\n\n")

    peaks_file.write("Number    Wavelength \n")
    for i_peak in range(len(pu_peaks_x)):
        curr_peak = pu_peaks_x[i_peak]
        peaks_file.write(str(i_peak) + "  " + str(curr_peak) + "\n")

    # manually selecting which peak is the [OII] peak - given in wavelength
    if (pu_peaks_x.size != 0):
        otwo_wav    = float(pu_peaks_x[0])  
        otwo_acc    = float(sl['emis']['[OII]'])

        redshift = (otwo_wav / otwo_acc) - 1
    else:
        # accepting HST redshift if cannot find peak
        redshift = hst_redshift

    return {'gd_shifted': gd_mc, 'sky_noise': sn_data, 'spectra': sl, 'redshift': 
            redshift, 'pu_peaks': pu_peaks_x}

def find_nearest(array, value):
    """ Find nearest value is an array """
    idx = (np.abs(array-value)).argmin()
    return idx

def sn_line(x, c):
    return c

def sn_gauss(x, c, i1, mu, sigma1):
    norm = (sigma1*np.sqrt(2*np.pi))
    term1 = ( i1 / norm ) * np.exp(-(x-mu)**2/(2*sigma1**2))
    return (c + term1)

def chisq(y_model, y_data, y_err):
    csq = (y_data-y_model)**2 / y_err**2
    csq = np.sum(csq)

    red_csq = csq / (len(y_data) - 4)
    return {'chisq': csq, 'chisq_red': red_csq}

def sky_noise_weighting(file_name, sky_file_name):
    """ finding the sky noise from a small section of the cube data """
    cs_data     = spectra_analysis(file_name, sky_file_name)
    cube_data   = cs_data['gd_shifted']
    sn_data     = cs_data['sky_noise']
    wl_soln     = wavelength_solution(file_name)

    sn_data_min = np.min(sn_data)
    in_wt       = 1 / (sn_data - sn_data_min + 1)

    sky_regns = np.zeros((len(in_wt),2)) # storing regions of potential sky noise
    for i in range(len(in_wt)): 
        data_acl = cube_data[i]
        data_sky = sn_data[i]
        data_prb = in_wt[i]
         
        if ( 0.00 <= np.abs(data_prb) <= 1.00 ):
            sky_regns[i][0] = data_prb
            sky_regns[i][1] = data_sky

    # finding max peak in the sky-noise data and fitting a Gaussian to that
    # x-axis data
    x_range = np.linspace(wl_soln['begin'], wl_soln['end'], wl_soln['steps'])

    # Finding peaks with PeakUtils
    sky_peaks = peakutils.indexes(sn_data, thres=300, thres_abs=True)
    sky_peaks_x = peakutils.interpolate(x_range, sn_data, sky_peaks)

    if (sky_peaks_x.size != 0):
        sky_peak = sky_peaks_x[0]
        sky_peak_index = find_nearest(sky_peak, x_range)
    else:
        sky_peak = 6000
        sky_peak_index = 0

    sky_peak_loc = x_range[sky_peak_index]

    sky_peak_range = [sky_peak-100, sky_peak+100]
    sky_peak_range_loc = [find_nearest(x_range, x) for x in sky_peak_range]

    sky_rng_x = x_range[sky_peak_range_loc[0]:sky_peak_range_loc[1]]
    sky_rng_y = sn_data[sky_peak_range_loc[0]:sky_peak_range_loc[1]]

    sky_gauss_params  = Parameters()
    sky_gauss_params.add('c', value=0)
    sky_gauss_params.add('i1', value=np.max(sky_rng_y), min=0.0)
    sky_gauss_params.add('mu', value=sky_peak_loc)
    sky_gauss_params.add('sigma1', value=3)

    sky_gauss_model = Model(sn_gauss)
    sky_gauss_rslt  = sky_gauss_model.fit(sky_rng_y, x=sky_rng_x, 
            params=sky_gauss_params)
    sky_gauss_best  = sky_gauss_rslt.best_values

    sky_sigma = sky_gauss_best['sigma1']

    return {'inverse_sky': in_wt, 'sky_regions': sky_regns, 'sky_sigma': sky_sigma}

def f_doublet(x, c, i1, i2, sigma_gal, z, sigma_inst):
    """ function for Gaussian doublet """  
    dblt_mu = [3727.092, 3729.875] # the actual non-redshifted wavelengths
    l1 = dblt_mu[0] * (1+z)
    l2 = dblt_mu[1] * (1+z)

    sigma = np.sqrt(sigma_gal**2 + sigma_inst**2)

    norm = (sigma*np.sqrt(2*np.pi))
    term1 = ( i1 / norm ) * np.exp(-(x-l1)**2/(2*sigma**2))
    term2 = ( i2 / norm ) * np.exp(-(x-l2)**2/(2*sigma**2)) 
    return (c*x + term1 + term2)

def otwo_doublet_fitting(file_name, sky_file_name):
    sa_data     = spectra_analysis(file_name, sky_file_name)
    y_shifted   = sa_data['gd_shifted'] 
    orr         = wavelength_solution(file_name)
    sn_data     = sky_noise_weighting(file_name, sky_file_name)

    redshift = sa_data['redshift']

    # obtaining the OII range and region
    # lower and upper bound on wavelength range
    lower_lambda = (1+redshift)*3600
    upper_lambda = (1+redshift)*3750

    otr = [lower_lambda, upper_lambda]

    orr_x       = np.linspace(orr['begin'], orr['end'], orr['steps'])
    dt_region   = [find_nearest(orr_x, x) for x in otr]
    otwo_region = y_shifted[dt_region[0]:dt_region[1]]

    ot_x        = orr_x[dt_region[0]:dt_region[1]]

    otwo_max_loc    = np.argmax(otwo_region)
    otwo_max_val    = np.max(otwo_region)

    # standard deviation of a range before the peak 
    stdr_b          = 50
    stdr_e          = otwo_max_loc - 50
    stddev_lim      = [stdr_b, stdr_e]

    stddev_x        = ot_x[stddev_lim[0]:stddev_lim[1]]
    stddev_region   = otwo_region[stddev_lim[0]:stddev_lim[1]]
    stddev_val      = np.std(stddev_region) 
    
    # fitting a gaussian doublet model to the data 
    dblt_mu = [3727.092, 3729.875] # the actual non-redshifted wavelengths  

    dblt_val = ot_x[otwo_max_loc]

    dblt_rng = [dblt_val-20, dblt_val+20]
    dblt_rng = [find_nearest(orr_x, x) for x in dblt_rng]
    dblt_rng_vals = orr_x[dblt_rng[0]:dblt_rng[1]]

    dblt_rgn = y_shifted[dblt_rng[0]:dblt_rng[1]]

    rdst = sa_data['redshift']

    sky_weight = sn_data['inverse_sky']
    sky_weight = sky_weight[dt_region[0]:dt_region[1]]

    # the parameters we need are (c, i1, i2, sigma1, z)    
    p0 = [0, otwo_max_val, 1.3, 3, rdst]
    c, i_val1, r, sigma_gal, z = p0 

    sigma_sky = sn_data['sky_sigma']
 
    gss_pars = Parameters()
    gss_pars.add('c', value=c)
    gss_pars.add('i1', value=i_val1, min=0.0)
    gss_pars.add('r', value=r, min=0.5, max=1.5)
    gss_pars.add('i2', expr='i1/r', min=0.0)
    gss_pars.add('sigma_gal', value=sigma_gal)
    gss_pars.add('z', value=z)
    gss_pars.add('sigma_inst', value=sigma_sky, vary=False)

    gss_model = Model(f_doublet)
    gss_result = gss_model.fit(otwo_region, x=ot_x, params=gss_pars, 
            weights=sky_weight) 

    opti_pms = gss_result.best_values
    init_pms = gss_result.init_values
   
    # working out signal to noise now
    sn_line_parms   = Parameters()
    sn_line_parms.add('c', value=c)

    sn_line_model   = Model(sn_line)
    sn_line_rslt    = sn_line_model.fit(otwo_region, x=ot_x, params=sn_line_parms)
    sn_line_bpms    = sn_line_rslt.best_values
    sn_line_data    = sn_line_rslt.best_fit

    sn_gauss_parms  = Parameters()
    sn_gauss_parms.add('c', value=c)
    sn_gauss_parms.add('i1', value=i_val1, min=0.0)
    sn_gauss_parms.add('mu', value=dblt_val)
    sn_gauss_parms.add('sigma1', value=sigma_gal)

    sn_gauss_model  = Model(sn_gauss)
    sn_gauss_rslt   = sn_gauss_model.fit(otwo_region, x=ot_x, params=sn_gauss_parms)
    sn_gauss_bpms   = sn_gauss_rslt.best_values
    sn_gauss_data   = sn_gauss_rslt.best_fit 

    sn_line_csqs    = chisq(sn_line_data, otwo_region, stddev_val)
    sn_gauss_csqs   = chisq(sn_gauss_data, otwo_region, stddev_val)

    signal_noise    = np.sqrt(sn_line_csqs['chisq'] - sn_gauss_csqs['chisq'])

    # saving data to text files
    curr_file_name = file_name.split('.')
    curr_file_name = curr_file_name[0].split('/')
    stk_f_n = curr_file_name[-1]

    data_dir = 'cube_results/' + stk_f_n
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
  
    file_writer.analysis_complete(data_dir, stk_f_n, gss_result, init_pms, opti_pms, 
            sn_line_csqs, sn_gauss_csqs, signal_noise, sn_line_bpms, sn_line_data,
            sn_gauss_bpms, sn_gauss_data) 

    return {'range': otr, 'x_region': ot_x,'y_region': otwo_region, 'doublet_range': 
            dblt_rng_vals, 'std_x': stddev_x, 'std_y': stddev_region, 'lm_best_fit': 
            gss_result.best_fit, 'lm_best_param': gss_result.best_values, 
            'lm_init_fit': gss_result.init_fit, 'sn_line': sn_line_rslt.best_fit, 
            'sn_gauss': sn_gauss_rslt.best_fit}

def analysis(file_name, sky_file_name):
    """ Graphs and results from analysing the cube for OII spectra """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    curr_file_name = file_name.split('.')
    curr_file_name = curr_file_name[0].split('/')
    stk_f_n = curr_file_name[-1]
    data_dir = 'cube_results/' + stk_f_n
   
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    spectra_stacker(file_name)

    # one figure to rule them all
    main_fig = plt.figure(1)

    # calling data once will be enough
    im_coll_data = image_collapser(file_name)
    spectra_data = spectrum_creator(file_name)
    sr = wavelength_solution(file_name) 
    df_data = otwo_doublet_fitting(file_name, sky_file_name) 
    gs_data = spectra_analysis(file_name, sky_file_name)
    snw_data = sky_noise_weighting(file_name, sky_file_name)

    # --- for collapsed images ---
    def graphs_collapsed():  
        f, (ax1, ax2)  = plt.subplots(1, 2)
    
        ax1.imshow(im_coll_data['median'], cmap='gray_r') 
        ax1.set_title(r'\textbf{galaxy: median}', fontsize=13)    
        ax1.set_xlabel(r'\textbf{Pixels}', fontsize=13)
        ax1.set_ylabel(r'\textbf{Pixels}', fontsize=13) 

        ax2.imshow(im_coll_data['sum'], cmap='gray_r')
        ax2.set_title(r'\textbf{galaxy: sum}', fontsize=13)        
        ax2.set_xlabel(r'\textbf{Pixels}', fontsize=13)
        ax2.set_ylabel(r'\textbf{Pixels}', fontsize=13)
        
        f.subplots_adjust(wspace=0.4)
        f.savefig(data_dir + "/" + stk_f_n + '_collapsed_images.pdf')

    # --- spectra ---
    def graphs_spectra():
        f, (ax1, ax2)  = plt.subplots(2, 1) 

        # --- redshifted data plotting
        cbd_x   = np.linspace(sr['begin'], sr['end'], sr['steps'])
         
        ## plotting our cube data
        cbs_y   = gs_data['gd_shifted'] 
        ax1.plot(cbd_x, cbs_y, linewidth=0.5, color="#000000")

        ## plotting our sky noise data
        snd_y   = snw_data['sky_regions'][:,1]
        ax1.plot(cbd_x, snd_y, linewidth=0.5, color="#f44336", alpha=0.5)

        ## plotting our [OII] region
        ot_x    = df_data['x_region']
        ot_y    = df_data['y_region']
        ax1.plot(ot_x, ot_y, linewidth=0.5, color="#00c853") 

        ## plotting the standard deviation region in the [OII] section
        std_x   = df_data['std_x']
        std_y   = df_data['std_y']
        ax1.plot(std_x, std_y, linewidth=0.5, color="#00acc1") 
        
        ## plotting peak lines for scipy finder and peakutils finder
        #pk_lines = gs_data['gd_peaks']
        #for i in range(len(pk_lines)):
            #srb = sr['begin']
            #ax1.axvline(x=srb+pk_lines[i], linewidth=0.5, color="#8bc34a", alpha=0.2)
        
        pu_lines = gs_data['pu_peaks']
        for i in range(len(pu_lines)):
            srb = sr['begin']
            ax1.axvline(x=(pu_lines[i]), linewidth=0.5, color="#ec407a", alpha=0.2)

        ax1.set_title(r'\textbf{spectra: cross-section redshifted}', fontsize=13)    
        ax1.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
        ax1.set_ylabel(r'\textbf{Flux}', fontsize=13)
        ax1.set_ylim([-1000,5000]) # setting manual limits for now
    
        # --- corrected redshift
        crs_x   = np.linspace(sr['begin'], sr['end'], sr['steps'])
        rdst    = gs_data['redshift']

        sp_lines = gs_data['spectra']

        ## corrected wavelengths
        corr_x  = crs_x / (1+rdst)

        ## plotting our cube data
        cps_y   = gs_data['gd_shifted']
        ax2.plot(corr_x, cps_y, linewidth=0.5, color="#000000")

        ## plotting our sky noise data
        sn_y    =  gs_data['sky_noise']
        ax2.plot(corr_x, sn_y, linewidth=0.5, color="#e53935")
 
        ## plotting spectra lines
        for e_key, e_val in sp_lines['emis'].items():
            spec_line = float(e_val)
            ax2.axvline(x=spec_line, linewidth=0.5, color="#00c853")
            ax2.text(spec_line-10, 4800, e_key, rotation=-90)
 
        ax2.set_title(r'\textbf{spectra: cross-section corrected}', fontsize=13)        
        ax2.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
        ax2.set_ylabel(r'\textbf{Flux}', fontsize=13)
        ax2.set_ylim([-500,5000]) # setting manual limits for now

        f.subplots_adjust(hspace=0.5)
        f.savefig(data_dir + "/" + stk_f_n + '_spectra.pdf')

        # saving our plotting into npy files so they can be used elsewhere
        np.save(data_dir + "/" + stk_f_n + "_cbd_x", cbd_x)
        np.save(data_dir + "/" + stk_f_n + "_cbs_y", cbs_y)  

        np.save(data_dir + "/" + stk_f_n + "_snd_y", snd_y) 
        
        np.save(data_dir + "/" + stk_f_n + "_corr_x", corr_x)
        np.save(data_dir + "/" + stk_f_n + "_cps_y", cps_y)
    
    def graphs_otwo_region():
        ot_fig  = plt.figure(6)

        # plotting the data for the cutout [OII] region
        ot_x    = df_data['x_region']
        ot_y    = df_data['y_region']
        plt.plot(ot_x, ot_y, linewidth=1.5, color="#000000")

        ## plotting the standard deviation region in the [OII] section
        std_x   = df_data['std_x']
        std_y   = df_data['std_y']
        #plt.plot(std_x, std_y, linewidth=1.5, color="#00acc1") 

        dblt_rng    = df_data['doublet_range']
        ot_x_b, ot_x_e  = dblt_rng[0], dblt_rng[-1]
        x_ax_vals   = np.linspace(ot_x_b, ot_x_e, 1000)

        # lmfit 
        lm_init     = df_data['lm_init_fit']
        lm_best     = df_data['lm_best_fit'] 

        plt.plot(ot_x, lm_best, linewidth=1.5, color="#1e88e5", 
                label=r"\textbf{Best fit}")
        plt.plot(ot_x, lm_init, linewidth=1.5, color="#43a047", alpha=0.5,
                label=r"\textbf{Initial guess}")
        
        lm_params   = df_data['lm_best_param']
        lm_params   = [prm_value for prm_key, prm_value in lm_params.items()]
        c, i_val1, i_val2, sig_g, rdsh, sig_i = lm_params

        dblt_mu = [3727.092, 3729.875] # the actual non-redshifted wavelengths for OII
        l1 = dblt_mu[0] * (1+rdsh)
        l2 = dblt_mu[1] * (1+rdsh)
        
        sig = np.sqrt(sig_g**2 + sig_i**2) 
        norm = (sig*np.sqrt(2*np.pi))

        lm_y1 = c + ( i_val1 / norm ) * np.exp(-(ot_x-l1)**2/(2*sig**2))
        lm_y2 = c + ( i_val2 / norm ) * np.exp(-(ot_x-l2)**2/(2*sig**2))
    
        plt.plot(ot_x, lm_y1, linewidth=1.5, color="#e64a19", alpha=0.7, 
                label=r"\textbf{Gaussian 1}") 
        plt.plot(ot_x, lm_y2, linewidth=1.5, color="#1a237e", alpha=0.7,
                label=r"\textbf{Gaussian 2}")

        # plotting signal-to-noise straight line and gaussian to verify it works
        sn_line     = df_data['sn_line']
        sn_gauss    = df_data['sn_gauss']

        #plt.axhline(y=sn_line, linewidth=0.5, color="#5c6bc0", alpha=0.7) 
        #plt.plot(ot_x, sn_gauss, linewidth=0.5, color="#5c6bc0", alpha=0.7)

        #plt.title(r'\textbf{[OII] region}', fontsize=13)
        plt.legend(loc='upper left', prop={'size': 15})
        plt.tick_params(labelsize=20)
        plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=20)
        plt.ylabel(r'\textbf{Flux}', fontsize=20)
        plt.xlim([l1-100,np.max(ot_x)])
        plt.ylim([-100,np.max(ot_y)+100]) # setting manual limits for now
        plt.savefig(data_dir + "/" + stk_f_n + '_otwo_region.pdf',bbox_inches="tight")

    #graphs_collapsed()
    #graphs_spectra()
    graphs_otwo_region()

    plt.close("all")

    return {'image_data': im_coll_data, 'spectra_data': spectra_data, 'sr': sr,
            'df_data': df_data, 'gs_data': gs_data, 'snw_data': snw_data}

if __name__ == '__main__':
    analysis("/Volumes/Jacky_Cao/University/level4/project/cubes_better/" + 
        "cube_1068.fits", "data/skyvariance_csub.fits")
