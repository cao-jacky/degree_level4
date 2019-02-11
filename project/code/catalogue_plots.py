import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def read_cat(file_name):
    """ reads file_name and returns specific header data and image data """
    fits_file = fits.open(file_name)

    # Index 3 of the combined catalogue contains the combined tables of HST and from gaia
    header = fits_file[3].header
    data = fits_file[3].data
    return {'header': header, 'data': data}

def graph_sn_mag(x_data, y_data, cubes_data):
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, s=7, color="#000000", alpha=0.25)

    z_limit = (4800/3727) - 1 
    small_redshift = cubes_data[cubes_data[:,7]<=z_limit, :]
    sr_sn = small_redshift[:,9] / small_redshift[:,10]
    ax.scatter(small_redshift[:,5], sr_sn, s=7, color="#d50000", alpha=0.25)

    large_redshift = cubes_data[cubes_data[:,7]>=z_limit, :]
    catalogue = large_redshift[large_redshift[:,8].argsort()]
    catalogue = catalogue[0:300]
    cat_sn = catalogue[:,9] / catalogue[:,10]
    ax.scatter(catalogue[:,5], cat_sn, s=7, color="#00c853", alpha=0.25)

    cube_ids = catalogue[:,0]
    for i, txt in enumerate(cube_ids):
        pass
        #ax.annotate(int(txt), (catalogue[:,5][i], cat_sn[i]), alpha=0.2)

    ax.tick_params(labelsize=20)
    ax.set_xlabel(r'\textbf{HST V-band magnitude}', fontsize=20)
    ax.set_ylabel(r'\textbf{MUSE Image Flux S/N}', fontsize=20)

    ax.set_yscale('log')
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig("graphs/image_sn_vs_vband.pdf")
    plt.close("all")

def graph_counts_mag(x_data, y_data):
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, s=7, color="#000000")

    ax.tick_params(labelsize=20)
    ax.set_xlabel(r'\textbf{HST V-band magnitude}', fontsize=20)
    ax.set_ylabel(r'\textbf{MUSE Counts}', fontsize=20)

    ax.set_yscale('log')
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig("graphs/image_counts_vs_vband.pdf")
    plt.close("all")

def catalogue_analysis(file_name):
    file_read = read_cat(file_name)

    file_header = file_read['header']
    file_data = file_read['data']
 
    # I want to run through the entire catalogue, storing various things into indexes:
    # [0]: cube ID as determined by sextractor (375)
    # [1]: x-posn from image (376)
    # [2]: y-posn from image (377)
    # [3]: RA from image (381)
    # [4]: Dec from image (382)
    # [5]: HST 606nm mag (64)
    # [6]: HST 775nm mag (66)
    # [7]: redshift from MUSE (81)
    # [8]: probability of being a star (386)
    # [9]: flux from image (378)
    # [10]: flux_err from image (379)
    # [11]: isoarea from image (381)
    # [12]: OII3726 flux sum from catalogue (275)
    # [13]: OII3729 flux sum from catalogue (288)
    # [14]: HST 435nm mag (62)
    # [15]: HST 814nm mag (70)

    cubes_data = np.zeros((len(file_data), 16))

    for i_object in range(len(file_data)):
        curr_object = file_data[i_object]

        cubes_data[i_object][0] = curr_object[375] # cube id

        cubes_data[i_object][1] = curr_object[376] # image x-posn
        cubes_data[i_object][2] = curr_object[377] # image y-posn

        cubes_data[i_object][3] = curr_object[381] # image RA 
        cubes_data[i_object][4] = curr_object[382] # image Dec

        cubes_data[i_object][5] = curr_object[64] # HST 606nm mag
        cubes_data[i_object][6] = curr_object[66] # HST 775nm mag

        cubes_data[i_object][7] = curr_object[81] # MUSE redshift
        cubes_data[i_object][8] = curr_object[386] # probability of being a star
        cubes_data[i_object][9] = curr_object[378] # image flux
        cubes_data[i_object][10] = curr_object[379] # image flux error
        cubes_data[i_object][11] = curr_object[381] # image isoarea

        cubes_data[i_object][12] = curr_object[275] # OII3726 flux sum
        cubes_data[i_object][13] = curr_object[288] # OII3729 flux sum

        cubes_data[i_object][14] = curr_object[62] # HST 435nm mag
        cubes_data[i_object][15] = curr_object[70] # HST 814nm mag

    np.save("data/matched_catalogue_complete", cubes_data)

    # plotting all objects from image against HST catalogue
    flux = cubes_data[:,9]
    flux_err = cubes_data[:,10]
    flux_sn = flux / flux_err

    v_mag = cubes_data[:,5]

    graph_sn_mag(v_mag, flux_sn, cubes_data)
    graph_counts_mag(v_mag, flux)

    # sort by redshift z then cut the objects below the value of 0.3 
    z_limit = (4800/3727) - 1 

    #cubes_data = cubes_data[cubes_data[:,7].argsort()]
    cubes_data = cubes_data[cubes_data[:,7]>=z_limit, :]
    
    np.save("data/matched_catalogue", cubes_data)

    # attempting to deal with the star probability
    #cubes_data = cubes_data[cubes_data[:,8].argsort()[::-1]]
    #for i in range(len(cubes_data)):
        #print(cubes_data[i][0], cubes_data[i][8])


catalogue_analysis("data/matched_catalogues.fits")

