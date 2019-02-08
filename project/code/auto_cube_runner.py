import numpy as np

import cube_reader
import ppxf_fitter
import ppxf_fitter_kinematics_sdss
import voronoi_2d_binning
import spectra_data

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from lmfit import Parameters, Model

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
        print("Working with cube_"+str(cube_id))
        voronoi_2d_binning.voronoi_binning(cube_id)

def voronoi_plotter(cube_id): 
    vb_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_binned.npy") # Voronoi binned data

    ppxf_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_curr_voronoi_ppxf_results.npy") # pPXF data
    lmfit_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_voronoi_lmfit_results.npy") # lmfit data

    oc_data = np.load("data/cubes_better/cube_"+str(int(cube_id))+".npy")  

    # changing the shape of the data
    binned_data = np.zeros([np.shape(oc_data)[2],np.shape(oc_data)[1]])

    ppxf_sigma_data = np.copy(binned_data) # producing a copy for pPXF sigma data
    ppxf_vel_data = np.copy(binned_data) # copying for pPXF velocity data

    lmfit_sigma_data = np.copy(binned_data)
    lmfit_vel_data = np.copy(binned_data)

    curr_row = 0 
    for i_x in range(np.shape(oc_data)[2]):
        for i_y in range(np.shape(oc_data)[1]):
            vb_id = vb_data[curr_row][2]
            binned_data[i_y][i_x] = vb_id

            ppxf_loc = np.where(ppxf_data[:,1] == vb_id)[0]
            ppxf_vars = ppxf_data[ppxf_loc][0]
            
            ppxf_curr_vel = ppxf_vars[2]
            ppxf_curr_sigma = ppxf_vars[3]

            ppxf_vel_data[i_y][i_x] = ppxf_curr_vel
            ppxf_sigma_data[i_y][i_x] = ppxf_curr_sigma 

            lmfit_loc = np.where(lmfit_data[:,1] == vb_id)[0]
            lmfit_vars = lmfit_data[lmfit_loc][0]
            
            lmfit_curr_vel = lmfit_vars[2]
            lmfit_curr_sigma = lmfit_vars[3]

            lmfit_vel_data[i_y][i_x] = lmfit_curr_vel
            lmfit_sigma_data[i_y][i_x] = lmfit_curr_sigma         

            curr_row += 1

    ppxf_vel_unique = np.unique(ppxf_vel_data)
    ppxf_vel_data[ppxf_vel_data == 0] = np.nan

    ppxf_sigma_unique = np.unique(ppxf_sigma_data)
    ppxf_sigma_data[ppxf_sigma_data == 0] = np.nan

    lmfit_vel_unique = np.unique(lmfit_vel_data)
    lmfit_vel_data[lmfit_vel_data == 0] = np.nan

    lmfit_sigma_unique = np.unique(lmfit_sigma_data)
    lmfit_sigma_data[lmfit_sigma_data == 0] = np.nan


    f, (ax1, ax2) = plt.subplots(1,2)
    fax1 = ax1.imshow(np.fliplr(np.rot90(ppxf_vel_data,3)), cmap='jet', 
            vmin=ppxf_vel_unique[1], vmax=ppxf_vel_unique[-1])
    ax1.tick_params(labelsize=13)
    ax1.set_title(r'\textbf{Velocity Map}', fontsize=13)
    f.colorbar(fax1, ax=ax1)

    fax2 = ax2.imshow(np.fliplr(np.rot90(ppxf_sigma_data,3)), cmap='jet',
            vmin=ppxf_sigma_unique[1], vmax=ppxf_sigma_unique[-1])
    ax2.tick_params(labelsize=13)
    ax2.set_title(r'\textbf{Velocity Dispersion Map}', fontsize=13)
    f.colorbar(fax2, ax=ax2)

    f.tight_layout()
    f.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
            +"_ppxf_maps.pdf")

    g, (ax3, ax4) = plt.subplots(1,2)
    gax3 = ax3.imshow(np.fliplr(np.rot90(lmfit_vel_data,3)), cmap='jet', 
            vmin=ppxf_vel_unique[1], vmax=ppxf_vel_unique[-1])
    ax3.tick_params(labelsize=13)
    ax3.set_title(r'\textbf{Velocity Map}', fontsize=13)
    g.colorbar(gax3, ax=ax3)

    gax4 = ax4.imshow(np.fliplr(np.rot90(lmfit_sigma_data,3)), cmap='jet',
            vmin=ppxf_sigma_unique[1], vmax=ppxf_sigma_unique[-1])
    ax4.tick_params(labelsize=13)
    ax4.set_title(r'\textbf{Velocity Dispersion Map}', fontsize=13)
    g.colorbar(gax4, ax=ax4)

    g.tight_layout()
    g.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
            +"_lmfit_maps.pdf")

def voronoi_runner():
    # Running to obtain results from pPXF and OII fitting
    cf = ppxf_fitter.cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    uc = ppxf_fitter.usable_cubes(catalogue, bright_objects) # usable cubes
    uc = np.array([1578])
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

        # Array which stores cube_id, VorID, lmfit vel, and lmfit sigma (converted)
        cube_lmfit_results = np.zeros([len(voronoi_unique),4])
        
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

            # fitting OII doublet for the final spectra
            # wavelength solution
            x_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                    "_cbd_x.npy")

            # loading redshift and sigma_inst
            doublet_params = spectra_data.lmfit_data(cube_id)
            z = doublet_params['z']
            sigma_inst = doublet_params['sigma_inst']

            # masking out doublet region
            x_mask = ((x_data > (1+z)*3600) & (x_data < (1+z)*3750))
            x_masked = x_data[x_mask]
            y_masked = spectra[x_mask]

            oii_doublets = [3727.092, 3729.875]

            dbt_params = Parameters()
            dbt_params.add('c', value=0)
            dbt_params.add('i1', value=np.max(y_masked), min=0.0)
            dbt_params.add('r', value=1.3, min=0.5, max=1.5)
            dbt_params.add('i2', expr='i1/r', min=0.0)
            dbt_params.add('sigma_gal', value=3)
            dbt_params.add('z', value=z)
            dbt_params.add('sigma_inst', value=sigma_inst, vary=False)

            dbt_model = Model(spectra_data.f_doublet)
            dbt_result = dbt_model.fit(y_masked, x=x_masked, params=dbt_params)

            best_result = dbt_result.best_values
            best_z = best_result['z']
            best_sigma = best_result['sigma_gal']

            c = 299792.458 # speed of light in kms^-1
            lmfit_vel = c*np.log(1+best_z)

            lmfit_sigma = (best_sigma / (3727*(1+best_z))) * c
            
            # indexing data into lmfit array
            cube_lmfit_results[i_vid][0] = int(cube_id)
            cube_lmfit_results[i_vid][1] = int(i_vid)
            cube_lmfit_results[i_vid][2] = lmfit_vel
            cube_lmfit_results[i_vid][3] = lmfit_sigma             
            
        # Save each cube_ppxf_results into cube_results folder
        np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                "_voronoi_ppxf_results.npy", cube_ppxf_results)

        # saving cube_lmfit_results into cube_results folder
        np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                "_voronoi_lmfit_results.npy", cube_lmfit_results)



if __name__ == '__main__':
    #voronoi_cube_runner()
    voronoi_runner()
    voronoi_plotter(1578)
