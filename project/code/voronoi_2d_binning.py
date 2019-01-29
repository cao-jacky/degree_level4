#!/usr/bin/env python

import os 

from os import path
import numpy as np
import matplotlib.pyplot as plt

import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

#-----------------------------------------------------------------------------

def data_file_creator(cube_id):
    # I need to create four columns: x-coord, y-coord, signal, noise
    # load up the inidividual MUSE cubes, then save the data as a numpy array file?
    cube_data = ("data/cubes_better/cube_"+str(int(cube_id))+".npy")
    if not (os.path.exists(cube_data)):
        print("test")
        pass
    else:
        pass

    # create array which has rows equal to x*y 

    # read over every pixel and store the signal, noise into the array

def voronoi_binning(cube_id):

    file_dir = path.dirname(path.realpath(vorbin.__file__))  # path of vorbin
    x, y, signal, noise = np.loadtxt(file_dir + '/voronoi_2d_binning_example_input.txt').T
    targetSN = 50.0

    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    #
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
        x, y, signal, noise, targetSN, plot=1, quiet=0)

    # Save to a text file the initial coordinates of each pixel together
    # with the corresponding bin number computed by this procedure.
    # binNum uniquely specifies the bins and for this reason it is the only
    # number required for any subsequent calculation on the bins.
    #
    np.savetxt('voronoi_2d_binning_example_output.txt', np.column_stack([x, y, binNum]),
               fmt=b'%10.6f %10.6f %8i')

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    data_file_creator(1804)

    #voronoi_binning_example()
    #plt.tight_layout()
    #plt.pause(1)

    
