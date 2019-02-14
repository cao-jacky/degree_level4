import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def getstats(img,ff): 
    gd = np.where(img != 0)
    print('there are ',len(gd),' elements')
    arr = img[gd]
    arr = sorted(arr)
    n = len(arr)
    print('array is ',n,' elements')
    i = round(ff*n)
    vmax = arr[i]
    print(ff,' signal range value is ',vmax)
    
    print('making mask')
    mask = make_source_mask(img, snr=2, npixels=5, dilate_size=11)
    print('calculating stats')
    vmean, vmedian, vstd = sigma_clipped_stats(img, sigma=3.0, mask=mask,mask_value=0.)
    print('mean: ',vmean)
    print('median: ',vmedian)
    print('sigma: ',vstd)
    return vmean,vmedian,vstd,vmax

def mkcol(b,v,r,ff,gamma):
    bmean,bmedian,bstd,bmax = getstats(b,ff)
    vmean,vmedian,vstd,vmax = getstats(v,ff)
    rmean,rmedian,rstd,rmax = getstats(r,ff)

    bmin = bmean
    vmin = vmean
    rmin = rmean

    gdb = np.where(b != 0)
    gdv = np.where(v != 0)
    gdr = np.where(r != 0)

    b[gdb] = (b[gdb]-bmin)/(bmax-bmin)
    v[gdb] = (v[gdb]-vmin)/(vmax-vmin)
    r[gdb] = (r[gdb]-rmin)/(rmax-rmin)

    lo = 0.
    hi = 1.

    bad = np.where(b <= lo)
    b[bad]=0.
    bad = np.where(b >= hi)
    b[bad]=1.

    bad = np.where(v <= lo)
    v[bad]=0
    bad = np.where(v >= hi)
    v[bad]=1.

    bad = np.where(r <= lo)
    r[bad]=0
    bad = np.where(r >= hi)
    r[bad]=1.

    #np.save("data/hst_data/b.npy", b)
    #np.save("data/hst_data/v.npy", v)
    #np.save("data/hst_data/r.npy", r)

    b = b**gamma
    v = v**gamma
    r = r**gamma

#    b = b*254.
#    v = v*254.
#    r = r*254.

    sz = b.shape
    print(sz[1],sz[0])
    
    col = np.zeros((sz[0],sz[1],3))
    col[:,:,0] = b
    col[:,:,1] = v
    col[:,:,2] = r

    return col

def read_cube(file_name):
    fits_file = fits.open(file_name)
    data = fits_file[0].data
    return data

def colour_image():
    b = read_cube("/Volumes/Jacky_Cao/University/level4/project/HST_HUDF/hlsp_xdf_hst_acswfc-30mas_hudf_f435w_v1_sci.fits")
    v = read_cube("/Volumes/Jacky_Cao/University/level4/project/HST_HUDF/hlsp_xdf_hst_acswfc-30mas_hudf_f606w_v1_sci.fits")
    i = read_cube("/Volumes/Jacky_Cao/University/level4/project/HST_HUDF/hlsp_xdf_hst_acswfc-30mas_hudf_f775w_v1_sci.fits")

    data_array = np.zeros((b.shape[0], b.shape[1], 3), dtype=float)

    #image = make_lupton_rgb(b, v, i, Q=10, stretch=0.5, 
            #filename="results/hst_hudf.jpeg")

    colour_data = mkcol(b,v,i, 0.99, 0.5)

    fig = plt.figure()
    fig.set_size_inches(10,10)
    data_array[:,:,0] = b 
    data_array[:,:,1] = v 
    data_array[:,:,2] = i 
    plt.imshow(colour_data, interpolation='nearest', origin='lower')
    plt.axis('off')
    #plt.show()
    plt.savefig('results/hubble_ultra_deep_field.png', dpi=(500), bbox_inches='tight',
            pad_inches=0.0)

if __name__ == '__main__':
    colour_image()
