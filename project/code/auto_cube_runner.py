import numpy as np

import cube_reader
import ppxf_fitter
import voronoi_2d_binning

def voronoi_cube_runner():
    # producing voronoi plots and data areas
    cf = ppxf_fitter.cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    uc = ppxf_fitter.usable_cubes(catalogue, bright_objects) # usable cubes

    for i_cube in range(len(uc)):
        cube_id = int(uc[i_cube])
        voronoi_2d_binning.voronoi_binning(cube_id)

def voronoi_ppxf_runner():
    cf = ppxf_fitter.cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    uc = ppxf_fitter.usable_cubes(catalogue, bright_objects) # usable cubes
    uc = np.array([1804])
    for i_cube in range(len(uc)):
        cube_id = int(uc[i_cube])

        # loading the MUSE spectroscopic data
        file_name = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/" 
                    + "cube_" + str(cube_id) + ".fits")
        fits_file = cube_reader.read_file(file_name)
        image_data = fits_file[1]
    
        # open the voronoi binned data
        voronoi_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
                +"_binned.npy")
        voronoi_unique = np.unique(voronoi_data[:,2])
        
        for i_vid in range(len(voronoi_unique)):
            curr_vid = int(voronoi_unique[i_vid])
            
            # find what pixels are at the current voronoi id 
            curr_where = np.where(voronoi_data[:,2] == curr_vid)[0]

            # create a single spectra from the found pixels
            spectra = np.zeros([np.shape(image_data)[0]])

            if len(curr_where) == 1:
                pixel_x = int(voronoi_data[curr_where][0][0])
                pixel_y = int(voronoi_data[curr_where][0][1])

                single_spec = image_data[:][:,pixel_y][:,pixel_x]
                spectra = spectra + single_spec 
            else: 
                for i_cw in range(len(curr_where)):
                    curr_pixel_id = curr_where[i_cw]

                    pixel_x = int(voronoi_data[curr_pixel_id][0])
                    pixel_y = int(voronoi_data[curr_pixel_id][1])
                    
                    curr_spec = image_data[:][:,pixel_y][:,pixel_x]
                    spectra = spectra + curr_spec

            # run pPXF on the final spectra and store results
    
if __name__ == '__main__':
    #cube_runner()
    voronoi_ppxf_runner()
