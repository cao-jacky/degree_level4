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

import os

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

    sn_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                "_curr_voronoi_sn_results.npy") # signal to noise data

    oc_data = np.load("data/cubes_better/cube_"+str(int(cube_id))+".npy")  
    
    # Array to store various maps 
    # [0] : pPXF stellar velocity map
    # [1] : pPXF stellar velocity dispersion map
    # [2] : lmfit gas velocity map
    # [3] : lmfit gas velocity dispersion map
    # [4] : S/N map
    binned_data = np.zeros([5, np.shape(oc_data)[1],np.shape(oc_data)[0]])
   
    # obtaining redshift from integrated galaxy lmfit data
    lmfit_fitting = spectra_data.lmfit_data(cube_id)
    z = lmfit_fitting['z']

    # calculating velocity of galaxy based on redshift from integrated spectrum
    c = 299792.458 # speed of light in kms^-1
    vel_gal = c*np.log(1+z) # galaxy velocity

    curr_row = 0 
    for i_x in range(np.shape(oc_data)[0]):
        for i_y in range(np.shape(oc_data)[1]):
            vb_id = vb_data[curr_row][2]
            #binned_data[i_y][i_x] = vb_id

            # pPXF variables
            ppxf_loc = np.where(ppxf_data[:,1] == vb_id)[0]
            ppxf_vars = ppxf_data[ppxf_loc][0]
            
            binned_data[0][i_y][i_x] = ppxf_vars[2] - vel_gal # rest velocity
            binned_data[1][i_y][i_x] = ppxf_vars[3] # velocity dispersion 

            # lmfit variables
            lmfit_loc = np.where(lmfit_data[:,1] == vb_id)[0]
            lmfit_vars = lmfit_data[lmfit_loc][0]
            
            binned_data[2][i_y][i_x] = lmfit_vars[2] - vel_gal # rest velocity
            binned_data[3][i_y][i_x] = lmfit_vars[3] # velocity dispersion

            # S/N variable
            sn_loc = np.where(sn_data[:,1] == vb_id)[0]
            sn_vars = sn_data[sn_loc][0]

            binned_data[4][i_y][i_x] = sn_vars[2] # current signal-to-noise

            curr_row += 1

    # rotate the maps and save them as a numpy array instead of during imshow plotting
    ppxf_vel_data = np.fliplr(np.rot90(binned_data[0],3))
    ppxf_sigma_data = np.fliplr(np.rot90(binned_data[1],3))

    lmfit_vel_data = np.fliplr(np.rot90(binned_data[2],3))
    lmfit_sigma_data = np.fliplr(np.rot90(binned_data[3],3))

    curr_sn_data = np.fliplr(np.rot90(binned_data[4],3))

    ppxf_vel_unique = np.unique(ppxf_vel_data)
    ppxf_vel_data[ppxf_vel_data == 0] = np.nan

    ppxf_sigma_unique = np.unique(ppxf_sigma_data)
    ppxf_sigma_data[ppxf_sigma_data == 0] = np.nan

    lmfit_vel_unique = np.unique(lmfit_vel_data)
    lmfit_vel_data[lmfit_vel_data == 0] = np.nan

    lmfit_sigma_unique = np.unique(lmfit_sigma_data)
    lmfit_sigma_data[lmfit_sigma_data == 0] = np.nan

    # loading the binary segmentation map 
    seg_map = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
            str(int(cube_id))+"_segmentation.npy")

    f, (ax1, ax2) = plt.subplots(1,2)
    fax1 = ax1.imshow(ppxf_vel_data*seg_map, cmap='jet', 
            vmin=ppxf_vel_unique[1], vmax=ppxf_vel_unique[-1])
    ax1.tick_params(labelsize=13)
    ax1.set_title(r'\textbf{Velocity Map}', fontsize=13)
    f.colorbar(fax1, ax=ax1)

    fax2 = ax2.imshow(ppxf_sigma_data*seg_map, cmap='jet',
            vmin=ppxf_sigma_unique[1], vmax=ppxf_sigma_unique[-1])
    ax2.tick_params(labelsize=13)
    ax2.set_title(r'\textbf{Velocity Dispersion Map}', fontsize=13)
    f.colorbar(fax2, ax=ax2)

    f.tight_layout()
    f.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
            +"_ppxf_maps.pdf")

    g, (ax3, ax4) = plt.subplots(1,2)
    gax3 = ax3.imshow(lmfit_vel_data*seg_map, cmap='jet', 
            vmin=ppxf_vel_unique[1], vmax=ppxf_vel_unique[-1])
    ax3.tick_params(labelsize=13)
    ax3.set_title(r'\textbf{Velocity Map}', fontsize=13)
    g.colorbar(gax3, ax=ax3)

    gax4 = ax4.imshow(lmfit_sigma_data*seg_map, cmap='jet',
            vmin=ppxf_sigma_unique[1], vmax=ppxf_sigma_unique[-1])
    ax4.tick_params(labelsize=13)
    ax4.set_title(r'\textbf{Velocity Dispersion Map}', fontsize=13)
    g.colorbar(gax4, ax=ax4)

    g.tight_layout()
    g.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
            +"_lmfit_maps.pdf")

    h, (ax5) = plt.subplots(1,1)
    hax5 = ax5.imshow(curr_sn_data*seg_map, cmap='jet', 
            vmin=np.min(sn_data[:,2]), vmax=np.max(sn_data[:,2]))
    ax5.tick_params(labelsize=13)
    ax5.set_title(r'\textbf{S/N Map}', fontsize=13)
    h.colorbar(hax5, ax=ax5)

    h.tight_layout()
    h.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
            +"_signal_noise_map.pdf")

def voronoi_runner():
    # Running to obtain results from pPXF and OII fitting
    cf = ppxf_fitter.cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    run_sn = True
    run_ppxf = False
    run_lmfit = False

    uc = ppxf_fitter.usable_cubes(catalogue, bright_objects) # usable cubes
    uc = np.array([1804])
    for i_cube in range(len(uc)):
        cube_id = int(uc[i_cube])

        l, (sax1) = plt.subplots(1,1) # spectra and pPXF on same plot
        m, (max1) = plt.subplots(1,1) # pPXF plots
        n, (nax1) = plt.subplots(1,1) # spectra plots

        # loading the MUSE spectroscopic data
        file_name = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/" 
                    +"cube_"+str(cube_id)+".fits")
        fits_file = cube_reader.read_file(file_name)

        image_data = fits_file[1] # image data from fits file is "wrong way around"
        image_data =  np.fliplr(np.rot90(image_data, 1, (1,2)))

        # loading Voronoi map - need to add 1 to distinguish between 1st bin and 
        # the 'off' areas as defined by binary segmentation map
        vor_map = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
                str(int(cube_id))+"_voronoi_map.npy") + 1

        # loading the binary segmentation map 
        seg_map = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
                str(int(cube_id))+"_segmentation.npy")

        vor_map = vor_map * seg_map
        voronoi_unique = np.unique(vor_map)
    
        # loading the wavelength solution
        ws_data = cube_reader.wavelength_solution(file_name)
        wl_sol = np.linspace(ws_data['begin'], ws_data['end'], ws_data['steps'])
    
        # open the voronoi binned data
        voronoi_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
                +"_binned.npy")
    
        # Array which stores cube_id, VorID, S/N for region
        cube_sn_results = np.zeros([len(voronoi_unique),3])

        # Array which stores cube_id, VorID, pPXF vel, and pPXF sigma
        cube_ppxf_results = np.zeros([len(voronoi_unique),4])

        # Array which stores cube_id, VorID, lmfit vel, and lmfit sigma (converted)
        cube_lmfit_results = np.zeros([len(voronoi_unique),4])

        # Applying the segmentation map to the Voronoi map
        # I want to work with the converted map 
        
        for i_vid in range(len(voronoi_unique)):
            if i_vid == 0:
                # ignoring the 'off' areas of the Voronoi map
                pass
            else:
                curr_vid = int(voronoi_unique[i_vid])
                print("Considering cube_"+str(cube_id)+" and Voronoi ID "+
                        str(curr_vid))
                
                # find what pixels are at the current voronoi id
                curr_where = np.where(vor_map == curr_vid)
                
                # create a single spectra from the found pixels
                spectra = np.zeros([len(curr_where[0]),np.shape(image_data)[0]])
                print(np.shape(spectra))
         
                if len(curr_where) == 1:
                    pixel_x = int(curr_where[0])
                    pixel_y = int(curr_where[1])

                    single_spec = image_data[:][:,pixel_x][:,pixel_y]
                    spectra[0] = single_spec
                else:
                    spec_counter = 0 
                    for i_x in range(len(curr_where[0])):
                        # looking at current x and y positions
                        pixel_x = int(curr_where[0][i_x]) 
                        pixel_y = int(curr_where[1][i_x]) 
                    
                        curr_spec = image_data[:][:,pixel_y][:,pixel_x]

                        # saving spectra into specific row of spectra array
                        spectra[spec_counter] = curr_spec 
                        
                        spec_counter += 1

                spectra = np.nansum(spectra, axis=0)

                # calculate the S/N on the new generated spectra
                # parameters from lmfit
                lm_params = spectra_data.lmfit_data(cube_id)
                z = lm_params['z']

                region = np.array([4000,4080]) * (1+z)
                region_mask = ((wl_sol > region[0]) & (wl_sol < region[1]))
                
                # masking useful region
                masked_wlr = wl_sol[region_mask]
                masked_spec = spectra[region_mask]

                signal = masked_spec
                noise = np.std(masked_spec) 

                signal_noise = np.abs(np.average(signal/noise))
                print(np.median(signal), noise, signal_noise)

                cube_sn_results[i_vid][0] = int(cube_id)
                cube_sn_results[i_vid][1] = int(i_vid)
                cube_sn_results[i_vid][2] = int(signal_noise)

                np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                    "_curr_voronoi_sn_results.npy", cube_sn_results)

                # run pPXF on the final spectra and store results 
                if np.isnan(np.sum(spectra)) == True:
                    ppxf_vel = 0
                    ppxf_sigma = 0
                else:
                    ppxf_run = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 
                            spectra, "all")
                    plt.close("all")
                    ppxf_vars = ppxf_run['variables']

                    ppxf_vel = ppxf_vars[0]
                    ppxf_sigma = ppxf_vars[1]

                    # use the returned data from pPXF to plot the spectra
                    x_data = ppxf_run['x_data']
                    y_data = ppxf_run['y_data']                
                    best_fit = ppxf_run['model_data']

                    # plot indidividual spectra 
                    indiv_spec_dir = ("cube_results/cube_"+str(cube_id)+
                            "/voronoi_spectra") 

                    if not os.path.exists(indiv_spec_dir):
                        os.mkdir(indiv_spec_dir)

                    t, (tax1) = plt.subplots(1,1)
                    tax1.plot(x_data, y_data, lw=1.5, c="#000000")
                    tax1.plot(x_data, best_fit, lw=1.5, c="#d32f2f") 
            
                    tax1.tick_params(labelsize=20)
                    tax1.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=20)
                    tax1.set_ylabel(r'\textbf{Relative Flux}', fontsize=20)

                    t.tight_layout()
                    t.savefig(indiv_spec_dir+"/cube_"+str(cube_id)+"_"+str(i_vid)+
                            "_spectra.pdf")
                    plt.close("all")
     
                    # plotting initial spectra
                    sax1.plot(x_data, y_data, lw=1.5, c="#000000")         
                    # plotting pPXF best fit
                    sax1.plot(x_data, best_fit, lw=1.5, c="#d32f2f") 

                    max1.plot(x_data, best_fit+150*i_vid, lw=1.5, c="#d32f2f")
                    nax1.plot(x_data, y_data+1000*i_vid, lw=1.5, c="#000000")

                # Storing data into cube_ppxf_results array
                cube_ppxf_results[i_vid][0] = int(cube_id)
                cube_ppxf_results[i_vid][1] = int(i_vid)
                cube_ppxf_results[i_vid][2] = ppxf_vel
                cube_ppxf_results[i_vid][3] = ppxf_sigma

                np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                    "_curr_voronoi_ppxf_results.npy", cube_ppxf_results)

                # fitting OII doublet for the final spectra
                # wavelength solution
                x_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+
                        str(cube_id)+"_cbd_x.npy")

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
           
        sax1.tick_params(labelsize=20)
        sax1.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=20)
        sax1.set_ylabel(r'\textbf{Relative Flux}', fontsize=20)
        l.tight_layout()
        l.savefig("cube_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))
                + "_voronoi_spectra_stacked.pdf")

        max1.tick_params(labelsize=20)
        max1.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=20)
        max1.set_ylabel(r'\textbf{Relative Flux}', fontsize=20)
        m.tight_layout()
        m.savefig("cube_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))
                + "_voronoi_spectra_stacked_ppxf.pdf") 

        nax1.tick_params(labelsize=20)
        nax1.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=20)
        nax1.set_ylabel(r'\textbf{Relative Flux}', fontsize=20)
        n.tight_layout()
        n.savefig("cube_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))
                + "_voronoi_spectra_stacked_spectra.pdf") 

        # Save each cube_ppxf_results into cube_results folder
        np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                "_voronoi_ppxf_results.npy", cube_ppxf_results)

        # saving cube_lmfit_results into cube_results folder
        np.save("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                "_voronoi_lmfit_results.npy", cube_lmfit_results)

def galaxy_rotator(cube_id):
    # load the velocity maps for stars and gas

    # rotate array by an angle

    # save the rotated array 

    # create an image
    pass

if __name__ == '__main__':
    #voronoi_cube_runner()
    #voronoi_runner()
    voronoi_plotter(1804)
