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

read_cube("/Volumes/Jacky_Cao/University/level4/project/DATACUBE_UDF-MOSAIC.fits")

    
