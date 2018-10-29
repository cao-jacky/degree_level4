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

def graph_sn_mag(x_data, y_data):
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, s=7, color="#000000")

    ax.set_xlabel(r'\textbf{HST V-bang magnitude}', fontsize=13)
    ax.set_ylabel(r'\textbf{MUSE Flux S/N}', fontsize=13)

    ax.set_yscale('log')
    ax.invert_xaxis()
    plt.savefig("graphs/sanity_checks/image_sn_vs_vband.pdf")
    plt.close("all")

def catalogue_analysis(file_name):
    file_read = read_cat(file_name)

    file_header = file_read['header']
    file_data = file_read['data']

    curr_cat_row = 0
    for l in zip(*file_data):
        if (curr_cat_row == 378):
            flux = np.array(l)
        if (curr_cat_row == 379):
            flux_err = np.array(l)
        if (curr_cat_row == 64):
            v_mag = np.array(l)
        curr_cat_row += 1
    
    flux_sn = flux / flux_err

    graph_sn_mag(v_mag, flux_sn)


catalogue_analysis("data/matched_catalogues.fits")

