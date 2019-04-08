import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from matplotlib import rc,rcParams
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def spectra_stacker():
    # list of all spectras
    spectral_types = {
            'O': '30614',
            'B': '17081',
            'A': '39866',
            'F': '5015',
            'G': 'G_7-6',
            'K': '5848',
            'M': 'G_176-11'
            }

    # create figure
    fig, axs = plt.subplots(len(spectral_types), 1, figsize=(16, 19))

    curr_st = 0 # counter for current spectral type
    for st_key, st_value in spectral_types.items():
        # load spectra fits file and the data within them
        fits_file = fits.open("data/comparison_spectra/"+st_value+".fits")
        fits_data = fits_file[1].data[0]

        axs[curr_st].plot(fits_data[0], fits_data[1], c="#000000", lw=1.5)
        axs[curr_st].text(9425, 0, r"\textbf{\textit{"+st_key+"}}", fontsize=40,
                color="#e53935")

        axs[curr_st].set_ylabel(r"\textbf{Flux}", fontsize=40)
        axs[curr_st].tick_params(labelsize=40)

        if curr_st != len(spectral_types)-1:
            axs[curr_st].get_xaxis().set_visible(False)

        curr_st += 1

    axs[len(spectral_types)-1].set_xlabel(r"\textbf{Wavelength (\AA)}", fontsize=40) 
    fig.savefig("graphs/spectral_types.pdf",bbox_inches="tight")
    plt.close("all")

if __name__ == '__main__':
    spectra_stacker()
