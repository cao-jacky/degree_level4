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
    data = fits_file[0].data
    return data

def colour_image():
    r = read_cube("data/hst/hlsp_hudf12_hst_wfc3ir_udfmain_f160w_v1.0_drz.fits")
    g = read_cube("data/hst/hlsp_hudf12_hst_wfc3ir_udfmain_f125w_v1.0_drz.fits")
    b = read_cube("data/hst/hlsp_hudf12_hst_wfc3ir_udfmain_f105w_v1.0_drz.fits")

    rgb_array = np.zeros((r.shape[0], r.shape[1], 3), dtype=float)

    fig = plt.figure()
    fig.set_size_inches(10,10)
    rgb_array[:,:,0] = r 
    rgb_array[:,:,1] = g 
    rgb_array[:,:,2] = b 
    plt.imshow(rgb_array, interpolation='nearest', origin='lower')
    plt.axis('off')
    plt.savefig('results/hubble_ultra_deep_field.png', dpi=(500), bbox_inches='tight',
            pad_inches=0.0)

colour_image()
