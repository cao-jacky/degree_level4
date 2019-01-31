import numpy as np

import cube_reader
import ppxf_fitter
import ppxf_fitter_kinematics_sdss
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

        # Array which stores cube_id, VorID, pPXF vel, and pPXF sigma
        cube_ppxf_results = np.zeros([len(voronoi_unique),4])
        
        for i_vid in range(len(voronoi_unique)):
            curr_vid = int(voronoi_unique[i_vid])
            print("Considering cube_"+str(cube_id)+" and Voronoi ID "+str(curr_vid))
            
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
            ppxf_run = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 
                    spectra, "all")
            ppxf_vars = ppxf_run['variables']

            # Storing data into cube_ppxf_results array
            cube_ppxf_results[i_vid][0] = int(cube_id)
            cube_ppxf_results[i_vid][1] = int(i_vid)
            cube_ppxf_results[i_vid][2] = ppxf_vars[0]
            cube_ppxf_results[i_vid][3] = ppxf_vars[1]

            np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                "_curr_voronoi_ppxf_results.npy", cube_ppxf_results)

        # Save each cube_ppxf_results into cube_results folder
        np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                "_voronoi_ppxf_results.npy", cube_ppxf_results)

if __name__ == '__main__':
    #cube_runner()
    voronoi_ppxf_runner()
