import numpy as np

import cube_reader
import ppxf_fitter
import ppxf_fitter_kinematics_sdss
import voronoi_2d_binning

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def voronoi_cube_runner():
    # producing voronoi plots and data areas
    cf = ppxf_fitter.cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    uc = ppxf_fitter.usable_cubes(catalogue, bright_objects) # usable cubes

    for i_cube in range(len(uc)):
        cube_id = int(uc[i_cube])
        voronoi_2d_binning.voronoi_binning(cube_id)

def voronoi_plotter(cube_id): 
    vb_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_binned.npy") # Voronoi binned data

    ppxf_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_voronoi_ppxf_results.npy") # pPXF data

    oc_data = np.load("data/cubes_better/cube_"+str(int(cube_id))+".npy")  

    # changing the shape of the data
    binned_data = np.zeros([np.shape(oc_data)[2],np.shape(oc_data)[1]])

    sigma_data = np.copy(binned_data) # producing a copy for sigma data
    vel_data = np.copy(binned_data) # copying for velocity data

    curr_row = 0 
    for i_x in range(np.shape(oc_data)[2]):
        for i_y in range(np.shape(oc_data)[1]):
            vb_id = vb_data[curr_row][2]
            binned_data[i_y][i_x] = vb_id

            ppxf_loc = np.where(ppxf_data[:,1] == vb_id)[0]
            ppxf_vars = ppxf_data[ppxf_loc][0]
            
            curr_vel = ppxf_vars[2]
            curr_sigma = ppxf_vars[3]

            vel_data[i_y][i_x] = curr_vel
            sigma_data[i_y][i_x] = curr_sigma 
            curr_row += 1

    vel_unique = np.unique(vel_data)
    vel_data[vel_data == 0] = np.nan

    sigma_unique = np.unique(sigma_data)
    sigma_data[sigma_data == 0] = np.nan

    print(sigma_unique)

    f, (ax1, ax2) = plt.subplots(1,2)
    fax1 = ax1.imshow(np.fliplr(np.rot90(vel_data,3)), cmap='jet', vmin=vel_unique[1], 
            vmax=vel_unique[-1])
    ax1.tick_params(labelsize=13)
    ax1.set_title(r'\textbf{Velocity Map}', fontsize=13)
    f.colorbar(fax1, ax=ax1)

    fax2 = ax2.imshow(np.fliplr(np.rot90(sigma_data,3)), cmap='jet')
    ax2.tick_params(labelsize=13)
    ax2.set_title(r'\textbf{Velocity Dispersion Map}', fontsize=13)
    f.colorbar(fax2, ax=ax2)

    f.tight_layout()
    f.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
            +"_ppxf_maps.pdf")

def voronoi_runner():
    # Running to obtain results from pPXF and OII fitting
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
            """
            if np.isnan(np.sum(spectra)) == True:
                ppxf_vel = 0
                ppxf_sigma = 0
            else:
                ppxf_run = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 
                        spectra, "all")
                ppxf_vars = ppxf_run['variables']

                ppxf_vel = ppxf_vars[0]
                ppxf_sigma = ppxf_vars[1]

            # Storing data into cube_ppxf_results array
            cube_ppxf_results[i_vid][0] = int(cube_id)
            cube_ppxf_results[i_vid][1] = int(i_vid)
            cube_ppxf_results[i_vid][2] = ppxf_vel
            cube_ppxf_results[i_vid][3] = ppxf_sigma

            np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                "_curr_voronoi_ppxf_results.npy", cube_ppxf_results)
            """

            # fit for the OII doublet for the final spectra

        # Save each cube_ppxf_results into cube_results folder
        #np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                #"_voronoi_ppxf_results.npy", cube_ppxf_results)



if __name__ == '__main__':
    #cube_runner()
    voronoi_runner()
    voronoi_plotter(1804)
