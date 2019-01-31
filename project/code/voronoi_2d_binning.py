#!/usr/bin/env python

import os 

from astropy.io import fits

from os import path
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

import cube_reader

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

#-----------------------------------------------------------------------------

def data_file_creator(cube_id):
    # I need to create four columns: x-coord, y-coord, signal, noise
    # load up the inidividual MUSE cubes, then save the data as a numpy array file?
    cube_data = ("data/cubes_better/cube_"+str(int(cube_id))+".npy")
    if not (os.path.exists(cube_data)):
        # obtaining the compressed median data for single galaxy cube
        file_name = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/" 
                    + "cube_" + str(cube_id) + ".fits")
        fits_file = cube_reader.read_file(file_name)

        image_data = fits_file[1]
        id_shape = np.shape(image_data)

        # 2 arrays to store the signal and then the noise
        cube_data_array = np.zeros((2,id_shape[1],id_shape[2]))
        for i_ra in range(id_shape[2]):
            for i_dec in range(id_shape[1]):
                pixel_data = image_data[:][:,i_dec][:,i_ra]

                # storing the sum pixel in signal array
                pd_sum = np.nansum(pixel_data)
                cube_data_array[0][i_dec][i_ra] = pd_sum

                # noise is the standard deviation
                pd_std = np.nanstd(pixel_data)
                cube_data_array[1][i_dec][i_ra] = pd_std 
        np.save(cube_data, cube_data_array)
        cube_data = cube_data_array
        pass
    else:
        cube_data = np.load(cube_data)
        pass 

    # array which has rows equal to x*y and four columns 
    cd_unpack = np.zeros([np.shape(cube_data)[1]*np.shape(cube_data)[2],4])
    cd_curr_row = 0
    # read over every pixel and store the signal, noise into the array
    for i_x in range(np.shape(cube_data)[2]):
        for i_y in range(np.shape(cube_data)[1]):
            cd_unpack[cd_curr_row][0] = i_x # x-coord
            cd_unpack[cd_curr_row][1] = i_y # y-corrd
            cd_unpack[cd_curr_row][2] = cube_data[0][i_y][i_x] # signal
            cd_unpack[cd_curr_row][3] = cube_data[1][i_y][i_x] # noise

            cd_curr_row += 1

    median_signal = np.nanmedian(cd_unpack[:,2])
    median_noise = np.nanmedian(cd_unpack[:,3])
    return {'data': cd_unpack, 'signal': median_signal, 'noise': median_noise}

def voronoi_binned_map(cube_id):
    cr_folder = "cube_results/cube_"+str(cube_id)
    vb_data = np.load(cr_folder+"/cube_"+str(cube_id)+"_binned.npy")

    # original cube_data
    o_cd = np.load("data/cubes_better/cube_"+str(int(cube_id))+".npy")    

    # changing the shape of the data
    binned_data = np.zeros([np.shape(o_cd)[2],np.shape(o_cd)[1]])
    curr_row = 0 
    for i_x in range(np.shape(o_cd)[2]):
        for i_y in range(np.shape(o_cd)[1]):
            binned_data[i_y][i_x] = vb_data[curr_row][2]
            curr_row += 1

    f, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(np.fliplr(np.rot90(binned_data,3)), cmap='hsv')
    ax1.tick_params(labelsize=13)

    ax2.imshow(np.fliplr(np.rot90(o_cd[0][:][:],3)))
    ax2.tick_params(labelsize=13)

    f.tight_layout()
    f.savefig(cr_folder+"/cube_"+str(cube_id)+"_voronoi_map_image.pdf")

    g, (gax1) = plt.subplots(1,1)
    gax1.imshow(np.fliplr(np.rot90(binned_data,3)), cmap='hsv')
    gax1.tick_params(labelsize=13)

    g.tight_layout()
    g.savefig(cr_folder+"/cube_"+str(cube_id)+"_voronoi_map.pdf")

def voronoi_binning(cube_id):
    cda = data_file_creator(cube_id) # cube_data_array
    cd = cda['data'] # cube_data

    x = cd[:,0]
    y = cd[:,1]
    signal = np.nan_to_num(cd[:,2])
    noise = np.nan_to_num(cd[:,3]) 

    for i_n in range(len(noise)):
        curr_noise = noise[i_n]
        if curr_noise == 0.0:
            noise[i_n] = 1.0

    targetSN = cda['signal'] / cda['noise']
    targetSN = 50

    print(targetSN)

    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
        x, y, signal, noise, targetSN, plot=1, quiet=0)
    
    binned = np.column_stack([x, y, binNum])
    cr_loc = ("cube_results/cube_"+str(cube_id)) 

    # saving to numpy array data file
    np.save(cr_loc+"/cube_"+str(cube_id)+"_binned.npy", binned)
    voronoi_binned_map(cube_id)

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    voronoi_binning(1804)

    #voronoi_binning_example()
    #plt.tight_layout()
    #plt.pause(1)

    
