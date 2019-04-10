import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

from astropy.io import fits

def seg_map_maker():
    # load the collapsed MUSE cube
    collapsed_muse = "data/IMAGE_UDF-MOSAIC.fits"
    cm_fits = fits.open(collapsed_muse)
    cm_data = cm_fits[1].data

    # load the segmentation map
    seg_map = "data/segmentation.fits"
    sm_fits = fits.open(seg_map)
    sm_data = sm_fits[0].data

    # plot the MUSE cube and segmentation map
    fig, ax = plt.subplots()
    ax.imshow(np.flipud(cm_data), cmap="binary", clim=(0.0, 0.995))
    #plt.hist(cm_data.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
     
    cmap = matplotlib.cm.prism  
    cmap.set_bad(color="#e0e0e0") # hiding background with no segmentation data
    sm_data = np.where(sm_data!=0, sm_data, np.nan)

    ax.imshow(np.flipud(sm_data), cmap="prism", alpha=0.6)

    ax.axis("off")
    fig.savefig("results/muse_seg_map.pdf", dpi=(500), bbox_inches="tight", 
            pad_inches=0.0)

if __name__ == '__main__':
    seg_map_maker()
