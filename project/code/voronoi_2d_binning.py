#!/usr/bin/env python

import os 

from astropy.io import fits

from os import path
import numpy as np

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
    if (os.path.exists(cube_data)):
        # obtaining the compressed median data for single galaxy cube
        file_name = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/" 
                    + "cube_" + str(cube_id) + ".fits")
        fits_file = cube_reader.read_file(file_name)

        image_data = fits_file[1]
        id_shape = np.shape(image_data)

        # loading segmentation data and turning everything not cube ID into 0 and
        # everything which *is* the cube ID into 1
        segmentation_data = fits_file[2]
        segmentation_data = np.fliplr(np.rot90(segmentation_data[:][:],3))

        segmentation_data[segmentation_data != cube_id] = 0
        segmentation_data[segmentation_data == cube_id] = 1

        np.save("cube_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))+
                "_segmentation.npy", segmentation_data)
        cb_noise = cube_reader.cube_noise(cube_id)
        cn_val = cb_noise['noise_value']

        # 2 arrays to store the signal and then the noise
        cube_data_array = np.zeros((id_shape[1],id_shape[2]))
        for i_ra in range(id_shape[2]):
            for i_dec in range(id_shape[1]):
                pixel_data = image_data[:][:,i_dec][:,i_ra]

                # storing the sum pixel in signal array
                pd_sum = np.nansum(pixel_data)
                cube_data_array[i_dec][i_ra] = pd_sum
        
        # rotated and flipped the data array
        cube_data_array = np.fliplr(np.rot90(cube_data_array[:][:],3))         
        np.save(cube_data, cube_data_array)
        cube_data = cube_data_array
        pass
    else:
        cube_data = np.load(cube_data)
        pass 
 
    cube_data_noise = np.std(cube_data[40:50,:]) # pixel noise from a non-data region
    cube_data = cube_data * segmentation_data # apply segmentation mask to data

    # array which has rows equal to x*y and four columns 
    cd_unpack = np.zeros((np.shape(cube_data)[0]*np.shape(cube_data)[1],4))
    cd_curr_row = 0
    # read over every pixel and store the signal, noise into the array
    for i_x in range(np.shape(cube_data)[0]):
        for i_y in range(np.shape(cube_data)[1]):
            cd_unpack[cd_curr_row][0] = i_x # x-coord
            cd_unpack[cd_curr_row][1] = i_y # y-corrd
            cd_unpack[cd_curr_row][2] = cube_data[i_x][i_y] # signal
            cd_unpack[cd_curr_row][3] = cube_data_noise

            cd_curr_row += 1

    median_signal = np.nanmedian(cd_unpack[:,2])
    median_noise = np.nanmedian(cd_unpack[:,3])
    return {'data': cd_unpack, 'original_data': cube_data, 
            'seg_map': segmentation_data}

def voronoi_binned_map(cube_id):
    cr_folder = "cube_results/cube_"+str(cube_id)
    vb_data = np.load(cr_folder+"/cube_"+str(cube_id)+"_binned.npy")

    # original cube_data
    o_cd = np.load("data/cubes_better/cube_"+str(int(cube_id))+".npy")    

    # segmentation data
    seg = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))+
                "_segmentation.npy")

    # changing the shape of the data
    binned_data = np.zeros([np.shape(o_cd)[0],np.shape(o_cd)[1]])
    curr_row = 0 
    for i_x in range(np.shape(o_cd)[0]):
        for i_y in range(np.shape(o_cd)[1]):
            binned_data[i_x][i_y] = vb_data[curr_row][2]
            curr_row += 1

    # save Voronoi map as an array
    np.save("cube_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))+
                "_voronoi_map.npy", binned_data)

    f, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(binned_data, cmap='prism') # plotting Voronoi tessellated map
    ax1.imshow(seg, cmap='gray', alpha=0.5) # overlaying segmentation map
    ax1.tick_params(labelsize=20)

    ax2.imshow(o_cd)
    ax2.tick_params(labelsize=20)

    f.tight_layout()
    f.savefig(cr_folder+"/cube_"+str(cube_id)+"_voronoi_map_image.pdf", 
            bbox_inches="tight")

    g, (gax1) = plt.subplots(1,1)
    gax1.imshow(binned_data, cmap='prism')
    gax1.tick_params(labelsize=20)
    #gax1.axis('off')

    g.tight_layout()
    g.savefig(cr_folder+"/cube_"+str(cube_id)+"_voronoi_map.pdf", bbox_inches="tight")

    #plt.show()

def voronoi_binning(cube_id):
    cda = data_file_creator(cube_id) # cube_data_array
    cd = cda['data'] # cube_data

    x = cd[:,0] * 0.20 # x and y have to be in arc seconds
    y = cd[:,1] * 0.20 # one MUSE pixel has size 0.2 arcsec/pixel
    signal = np.abs(np.nan_to_num(cd[:,2]))
    noise = cd[:,3] 

    targetSN = 40

    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
            x, y, signal, noise, targetSN, plot=1, quiet=0)

    #plt.show()
    
    binned = np.column_stack([x, y, binNum])
    cr_loc = ("cube_results/cube_"+str(cube_id)) 

    # saving to numpy array data file
    np.save(cr_loc+"/cube_"+str(cube_id)+"_binned.npy", binned)
    voronoi_binned_map(cube_id)

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    voronoi_binning(1804)
    #data_file_creator(1804)    
