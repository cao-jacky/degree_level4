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
from lmfit.models import StepModel, LinearModel

from scipy import ndimage

from astropy.io import ascii

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
    # [0] : pPXF stellar velocity map - de-redshifted
    # [1] : pPXF stellar velocity dispersion map
    # [2] : lmfit gas velocity map - redshifted
    # [3] : lmfit gas velocity dispersion map
    # [4] : S/N map
    # [5] : pPXF velocity errors map
    # [6] : lmfit velocity errors map
    # [7] : Voronoi ID map
    # [8] : pPXF stellar velocity map - redshifted
    # [9] : lmfit gas velocity map - redshifted
    binned_data = np.zeros([10, np.shape(oc_data)[1],np.shape(oc_data)[0]])
   
    # obtaining redshift from integrated galaxy lmfit data
    lmfit_fitting = spectra_data.lmfit_data(cube_id)
    z = lmfit_fitting['z']

    # calculating velocity of galaxy based on redshift from integrated spectrum
    c = 299792.458 # speed of light in kms^-1
    vel_gal = c*np.log(1+z) # galaxy velocity

    # velocity of galaxy for the central pixel
    cen_pix_vel_ppxf = ppxf_data[1][2]
    cen_pix_vel_lmfit = lmfit_data[1][2]

    # adding 1 to ignore the "0" bins which is area out of segmentation map
    vb_data[:,2] = vb_data[:,2] + 1

    curr_row = 0 
    for i_x in range(np.shape(oc_data)[0]):
        for i_y in range(np.shape(oc_data)[1]):
            vb_id = vb_data[curr_row][2]
            binned_data[7][i_y][i_x] = vb_id

            # S/N variable
            sn_loc = np.where(sn_data[:,1] == vb_id)[0]
            sn_vars = sn_data[sn_loc][0]

            binned_data[4][i_y][i_x] = sn_vars[2] # current signal-to-noise

            # pPXF variables and errors
            ppxf_loc = np.where(ppxf_data[:,1] == vb_id)[0]
            ppxf_vars = ppxf_data[ppxf_loc][0]
            
            binned_data[8][i_y][i_x] = ppxf_vars[2] # redshifted velocity
            binned_data[5][i_y][i_x] = ppxf_vars[4] # pPXF velocity error

            # lmfit variables
            lmfit_loc = np.where(lmfit_data[:,1] == vb_id)[0]
            lmfit_vars = lmfit_data[lmfit_loc][0]
            
            binned_data[9][i_y][i_x] = lmfit_vars[2] # redshifted velocity
            binned_data[6][i_y][i_x] = lmfit_vars[4] # lmfit velocity error 

            # storing values which are only high enough S/N
            if sn_vars[2] > 4:
                # rest velocities for pPXF and lmfit
                binned_data[0][i_y][i_x] = ppxf_vars[2] - cen_pix_vel_ppxf 
                binned_data[2][i_y][i_x] = lmfit_vars[2] - cen_pix_vel_lmfit
            if sn_vars[2] > 7:
                pass
            else:
                # velocity dispersions for pPXF and lmfit
                binned_data[1][i_y][i_x] = ppxf_vars[3] # velocity dispersion
                binned_data[3][i_y][i_x] = lmfit_vars[3] # velocity dispersion

            curr_row += 1

    # loading the binary segmentation map 
    seg_map = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
            str(int(cube_id))+"_segmentation.npy")

    # rotate the maps and save them as a numpy array instead of during imshow plotting
    binned_data = np.fliplr(np.rot90(binned_data, 1, (1,2))) 
    binned_data = binned_data * seg_map    

    np.save("cube_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))+
            "_maps.npy", binned_data)

    ppxf_vel_data = binned_data[0]
    ppxf_sigma_data = binned_data[1]

    lmfit_vel_data = binned_data[2]
    lmfit_sigma_data = binned_data[3]

    curr_sn_data = binned_data[4]

    ppxf_vel_unique = np.unique(ppxf_vel_data)
    ppxf_vel_data[ppxf_vel_data == 0] = np.nan

    ppxf_sigma_unique = np.unique(ppxf_sigma_data)
    ppxf_sigma_data[ppxf_sigma_data == 0] = np.nan

    lmfit_vel_unique = np.unique(lmfit_vel_data)
    lmfit_vel_data[lmfit_vel_data == 0] = np.nan

    lmfit_sigma_unique = np.unique(lmfit_sigma_data)
    lmfit_sigma_data[lmfit_sigma_data == 0] = np.nan

    curr_sn_data_unique = np.unique(curr_sn_data)
    curr_sn_data[curr_sn_data == 0] = np.nan
 
    # setting nan values to black
    #current_cmap = plt.cm.jet
    #current_cmap.set_bad(color='black')

    f, (ax1, ax2) = plt.subplots(1,2)
    fax1 = ax1.imshow(ppxf_vel_data, cmap='jet',
            vmin=ppxf_vel_unique[1], vmax=ppxf_vel_unique[-2])
    ax1.tick_params(labelsize=20)
    ax1.set_title(r'\textbf{Velocity Map}', fontsize=20)
    fcbar = f.colorbar(fax1, ax=ax1)
    fcbar.ax.tick_params(labelsize=20) 

    fax2 = ax2.imshow(ppxf_sigma_data, cmap='jet',
            vmin=ppxf_sigma_unique[1], vmax=ppxf_sigma_unique[-1])
    ax2.tick_params(labelsize=20)
    ax2.set_title(r'\textbf{Velocity Dispersion Map}', fontsize=20)
    fcbar = f.colorbar(fax2, ax=ax2)
    fcbar.ax.tick_params(labelsize=20) 

    f.tight_layout()
    f.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
            +"_ppxf_maps.pdf", bbox_inches="tight")

    g, (ax3, ax4) = plt.subplots(1,2)
    gax3 = ax3.imshow(lmfit_vel_data, cmap='jet', 
            vmin=ppxf_vel_unique[1], vmax=ppxf_vel_unique[-2])
    ax3.tick_params(labelsize=20)
    ax3.set_title(r'\textbf{Velocity Map}', fontsize=20)
    gcbar = g.colorbar(gax3, ax=ax3)
    gcbar.ax.tick_params(labelsize=20)

    gax4 = ax4.imshow(lmfit_sigma_data, cmap='jet',
            vmin=ppxf_sigma_unique[1], vmax=ppxf_sigma_unique[-1])
    ax4.tick_params(labelsize=20)
    ax4.set_title(r'\textbf{Velocity Dispersion Map}', fontsize=20)
    gcbar = g.colorbar(gax4, ax=ax4)
    gcbar.ax.tick_params(labelsize=20)

    g.tight_layout()
    g.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
            +"_lmfit_maps.pdf", bbox_inches="tight")

    h, (ax5) = plt.subplots(1,1)
    hax5 = ax5.imshow(curr_sn_data, cmap='jet', 
            vmin=np.min(sn_data[:,2]), vmax=np.max(sn_data[:,2]))
    
    #ax5.axis('off')
    ax5.tick_params(labelsize=20)
    #ax5.set_title(r'\textbf{S/N Map}', fontsize=20)

    hcbar = h.colorbar(hax5, ax=ax5)
    hcbar.ax.tick_params(labelsize=20)
    hcbar.ax.set_ylabel(r"$\bf{S/N}$", fontsize=20, rotation=270, labelpad=30) 

    h.tight_layout()
    h.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
            +"_signal_noise_map.pdf", bbox_inches="tight")

def voronoi_runner():
    # Running to obtain results from pPXF and OII fitting
    cf = ppxf_fitter.cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    uc = ppxf_fitter.usable_cubes(catalogue, bright_objects) # usable cubes
    #uc = uc[11:]
    uc = np.array([849])
    print(uc)
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

        # Array which stores cube_id, VorID, pPXF vel, pPXF sigma, vel err
        cube_ppxf_results = np.zeros([len(voronoi_unique),5])

        # Array which stores cube_id, VorID, lmfit vel, and lmfit sigma (converted)
        cube_lmfit_results = np.zeros([len(voronoi_unique),5])

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

                    # variables from pPXF
                    ppxf_vars = ppxf_run['variables']
                    ppxf_vel = ppxf_vars[0]
                    ppxf_sigma = ppxf_vars[1]

                    # errors from pPXF
                    ppxf_errs = ppxf_run['errors']
                    ppxf_vel_err = ppxf_errs[0]

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
                cube_ppxf_results[i_vid][4] = ppxf_vel_err

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

                lmfit_errors = dbt_result.params
                z_err = lmfit_errors['z'].stderr

                if z_err is None:
                    z_err = 0.0

                lmfit_vel_err = c*np.log(1+z_err)
                 
                # indexing data into lmfit array
                cube_lmfit_results[i_vid][0] = int(cube_id)
                cube_lmfit_results[i_vid][1] = int(i_vid)
                cube_lmfit_results[i_vid][2] = lmfit_vel
                cube_lmfit_results[i_vid][3] = lmfit_sigma             
                cube_lmfit_results[i_vid][4] = lmfit_vel_err  
           
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

        voronoi_plotter(cube_id)
        rotation_curves(cube_id)

def curve(x, a):
    return (a/x)

def rotation_curves(cube_id):
    # load the velocity maps for stars and gas
    galaxy_maps = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
            str(int(cube_id))+"_maps.npy")
    seg_map = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
            str(int(cube_id))+"_segmentation.npy")
    maps_list = {0: 'ppxf_velocity', 1: 'ppxf_velocity_dispersion', 
            2: 'lmfit_velocity', 3: 'lmfit_velocity_dispersion', 4: 'signal_noise',
            5: 'ppxf_vel_error', 6: 'lmfit_vel_error', 7: 'voronoi_id',
            8: 'ppxf_vel_redshifted', 9: 'lmfit_vel_redshifted'}

    # cutting all the galaxy maps according to velocity S/N or velocity dispersion S/N
    sn_map = galaxy_maps[4]

    # masking out regions in velocity maps with low S/N
    galaxy_maps[0] = np.where(sn_map > 4, galaxy_maps[0], -0.0)
    galaxy_maps[2] = np.where(sn_map > 4, galaxy_maps[2], -0.0)
    
    # obtaining fractional uncertainties from signal-to-noise
    # loading "a" factors in a/x model
    a_ppxf = np.load("uncert_ppxf/vel_curve_best_values_ppxf.npy")
    a_lmfit = np.load("uncert_lmfit/vel_curve_best_values_lmfit.npy")

    # Read the sextractor data file which contains various bits of info
    sextractor_data = np.loadtxt("data/GaiaCatalog0.ASC")
    sd_cc_loc = np.where(sextractor_data[:,0]==cube_id)[0] # current cube location

    # [8] : major axis in units of pixels
    # [9] : minor axis in units of pixels
    # [10] : angle between major axis and horizontal 
    sd_curr_cube = sextractor_data[sd_cc_loc][0]
    cc_b = sd_curr_cube[8] /2 # semi-major axis
    cc_a = sd_curr_cube[9] /2 # semi-major axis

    gal_inc = np.arccos(cc_a/cc_b) # inclination angle of galaxy in radians

    cc_ha = sd_curr_cube[10] # horizontal angle

    # scaling by pPXF maps
    ppxf_vel_data = galaxy_maps[0]
    ppxf_sigma_data = galaxy_maps[1]

    ppxf_vel_unique, ppxf_vu_counts = np.unique(ppxf_vel_data, return_counts=True)
    ppxf_sigma_unique = np.unique(ppxf_sigma_data)

    # I'm just going to do it manually
    cra = {
            '1804': 0,
            '1578': -180,
            '849': 90,
            '286': -190,
            '5': 45,
            '767': cc_ha + 10,
            '414': -25,
            '549': -100,
            '175': -55,
            '1129': 175,
            }

    if str(cube_id) in cra:
        rot_angle = cc_ha + cra[str(cube_id)] # rotation angle
    else:
        rot_angle = cc_ha

    # rotate all the maps by an angle and masking out non-useful regions
    galaxy_mask = np.where(galaxy_maps == 0, galaxy_maps, 1)
    rotated_galaxy_maps = ndimage.rotate(galaxy_maps, angle=rot_angle, axes=(1,2),
            mode='nearest', reshape=False)
 
    rotated_mask = ndimage.rotate(galaxy_mask, angle=rot_angle, axes=(1,2),
            mode='nearest', reshape=False)
    rotated_mask = np.where(rotated_mask > 1*10**(-1), rotated_mask, 0)
    
    # applying mask and setting 0 values to nan
    rotated_galaxy_maps = rotated_galaxy_maps * rotated_mask
    rotated_galaxy_maps = np.where(rotated_galaxy_maps != 0, rotated_galaxy_maps, 
            np.nan)
    
    # save the rotated arrays
    np.save("cube_results/cube_"+str(int(cube_id))+"/cube_"+
            str(int(cube_id))+"_rotated_maps.npy", rotated_galaxy_maps)

    g, (ax1) = plt.subplots(1,1) # 1D rotation curves for OII and gas
    p2, (p2ax1) = plt.subplots(1,1) # 2D  rotation curves for OII and gas

    # defining colours and labels
    rot_labels = {0: 'Stars', 2: 'Gas', 8: 'Stars', 9: 'Gas'}
    rot_c = {0: '#03a9f4', 2: '#f44336', 8: '#03a9f4', 9: '#f44336'}

    # array to store the following, sliced velocities
    # [0] : x-scale in units of pixel
    # [1] : pPXF velocity
    # [2] : lmfit velocity
    # [3] : pPXF velocity error
    # [4] : lmfit velocity error
    # [5] : S/N
    # [6] : Voronoi ID
    # [7] : pPXF velocity fractional error
    # [8] : lmfit velocity fractional error
    # [9] : x-scale in units of arcseconds
    sliced_vel = np.zeros([10, np.shape(rotated_galaxy_maps)[1]]) # original array
    sliced_vel = np.full_like(sliced_vel, np.nan, dtype=float) # fill with nan vals

    muse_scale = 0.20 # MUSE pixel scale in arcsec/pixel

    current_cmap = plt.cm.jet
    current_cmap.set_bad(color='white')

    # creating image for each map
    for i_map in range(np.shape(rotated_galaxy_maps)[0]):
        curr_map_data = rotated_galaxy_maps[i_map]
        map_string = maps_list[i_map]

        # finding central pixel
        map_shape = np.shape(curr_map_data)
        c_x = int(map_shape[0]/2)-1
        c_y = int(map_shape[1]/2)-1
        
        f, (ax) = plt.subplots(1,1) # converting the maps into a viewable pdf   

        # velocity maps
        if i_map in np.array([0,2]):
            # --------------------------------------------------#

            # working with rotated 2D data - major kinematic axis should be horizontal
            for cm_y in range(np.shape(curr_map_data)[1]):
                for cm_x in range(np.shape(curr_map_data)[0]):
                    cp_d = np.abs(curr_map_data[cm_y][cm_x]) # current pixel data 
                    
                    if np.isnan(cp_d) == True:
                        pass
                    else:
                        # current S/N for pixel
                        if (cm_x-c_x)**2/cc_b**2 + (cm_y-c_y)**2/cc_a**2 < 1:
                            cp_sn = rotated_galaxy_maps[4][cm_y][cm_x]                         
                            # top left corner is defined as (x=0,y=0) posn, 
                            # distances between central pixel and curr
                            cpx_dist = cm_x - c_x                 
                            cpy_dist = cm_y - c_y 
                            
                            cp_obs = cpx_dist**2 + cpy_dist**2

                            cp_radius = np.sqrt(cp_obs+cpy_dist**2*np.tan(gal_inc)**2)
                            cp_radius = cp_radius * muse_scale # convert to MUSE scale

                            if i_map == 0:
                                frac_err = a_ppxf # pPXF velocity fractional error
                                redshifted_cp_d = rotated_galaxy_maps[8][cm_y][cm_x]
                            else:
                                frac_err = a_lmfit # lmfit velocity fractional error
                                redshifted_cp_d = rotated_galaxy_maps[9][cm_y][cm_x]
                            y_err = curve(cp_sn, frac_err) * redshifted_cp_d

                            p2ax1.errorbar(cp_radius, cp_d, yerr=y_err, 
                                ms=5, fmt='o', c=rot_c[i_map], label=rot_labels[i_map],
                                elinewidth=1.0, capsize=5, capthick=1.0)
        
            # --------------------------------------------------#


            # slice containing the Voronoi IDs
            vid_slice = np.nanmedian(rotated_galaxy_maps[7][c_y-1:c_y+2,:], axis=0)
            vid_slice = np.nan_to_num(vid_slice)

            # unique Voronoi IDs and their locations 
            unique_vids, unique_locs = np.unique(vid_slice.astype(int), 
                    return_index=True)

            # select out a horizontal strip based on central pixel
            map_slice = curr_map_data[c_y-1:c_y+2,:]
            map_median = np.nanmedian(map_slice, axis=0)
            map_median = map_median[unique_locs] # masking out repeated values
            
            # extracting original velocity errors
            ppxf_vel_err_slice = np.nanmedian(rotated_galaxy_maps[5][c_y-1:c_y+2,:],
                    axis=0)[unique_locs]
            lmfit_vel_err_slice = np.nanmedian(rotated_galaxy_maps[6][c_y-1:c_y+2,:],
                    axis=0)[unique_locs]

            sliced_vel[3][0:len(ppxf_vel_err_slice)] = ppxf_vel_err_slice 
            sliced_vel[4][0:len(lmfit_vel_err_slice)] = lmfit_vel_err_slice

            # signal-to-noise for each bin
            sn_slice = np.nanmedian(rotated_galaxy_maps[4][c_y-1:c_y+2,:], 
                    axis=0)[unique_locs]

            sliced_vel[5][0:len(sn_slice)] = sn_slice  

            if i_map == 0:
                sliced_vel[1][0:len(map_median)] = map_median
                yerr = ppxf_vel_err_slice
            else:
                sliced_vel[2][0:len(map_median)] = map_median
                yerr = lmfit_vel_err_slice

            # array which defines the x-scale 
            x_scale = np.arange(0, map_shape[0], 1.0)  

            sliced_vel[0][0:len(x_scale[unique_locs])] = x_scale[unique_locs]

            x_scale = x_scale - c_x # setting central pixel as radius 0
            x_scale = x_scale * muse_scale # converting to MUSE scale
            x_scale = x_scale[unique_locs] # masking out repeated values

            sliced_vel[9][0:len(x_scale)] = x_scale 

            fax = ax.imshow(curr_map_data, cmap='jet', 
                    vmin=ppxf_vel_unique[1], vmax=ppxf_vel_unique[-2])  

            # overlaying area which has been considered
            #overlay_slice = curr_map_data * rotated_seg_map
            overlay_slice = curr_map_data
            overlay_slice[np.where(overlay_slice != 1.0)] = np.nan
            overlay_slice[c_y-1:c_y+2,:] = 2.0

            ax.imshow(overlay_slice, cmap='gray', alpha=0.5)
        # velocity dispersion maps
        if i_map in np.array([1,3]):
            fax = ax.imshow(curr_map_data, cmap='jet', 
                    vmin=ppxf_sigma_unique[1], vmax=ppxf_sigma_unique[-1]) 
        if i_map in np.array([8,9]):
            # slice containing the Voronoi IDs
            vid_slice = np.nanmedian(rotated_galaxy_maps[7][c_y-1:c_y+2,:], axis=0)
            vid_slice = np.nan_to_num(vid_slice)

            # unique Voronoi IDs and their locations 
            unique_vids, unique_locs = np.unique(vid_slice.astype(int), 
                    return_index=True)

            # select out a horizontal strip based on central pixel
            map_slice = curr_map_data[c_y-1:c_y+2,:]
            map_median = np.nanmedian(map_slice, axis=0)
            map_median = map_median[unique_locs] # masking out repeated values

            if i_map == 8:
                # pPXF velocity fractional error
                frac_err_ppxf = curve(sn_slice, a_ppxf) * map_median
                sliced_vel[7][0:len(sn_slice)] = frac_err_ppxf
                
                y_values = sliced_vel[1]
                y_err = sliced_vel[7]

            else:
                # lmfit velocity fractional error
                frac_err_lmfit = curve(sn_slice, a_lmfit) * map_median
                sliced_vel[8][0:len(sn_slice)] = frac_err_lmfit
                
                y_values = sliced_vel[2]
                y_err = sliced_vel[8] 

            x_values = sliced_vel[9]

            ax1.errorbar(x_values, y_values, yerr=y_err, 
                    ms=5, fmt='o', c=rot_c[i_map], 
                    label=rot_labels[i_map], elinewidth=1.0, capsize=5, capthick=1.0) 
        if i_map == 4:
            fax = ax.imshow(curr_map_data, cmap='jet')
       
        #ax.axis('off')
        ax.tick_params(labelsize=20)
        fcbar = f.colorbar(fax, ax=ax)
        fcbar.ax.tick_params(labelsize=20)

        if i_map in np.array([0]):
            fcbar.ax.set_ylabel(r"\textbf{Stellar velocity (kms$^{-1}$)}", 
                    fontsize=20, rotation=270, labelpad=30)

        f.tight_layout()
        if i_map not in np.array([5,6,7,8,9]):
            f.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                    "_rotated_"+map_string+".pdf", bbox_inches="tight") 

        plt.close("all")

    """
    # plotting model curve for rotation curve plot
    mcp = Parameters()
    mcp.add('sigma', value=0.0)
    mcp.add('center', value=0.0)
    mcp.add('amplitude', value=0.0)
    mcp.add('intercept', value=0.0)
    mcp.add('slope', value=1.0)

    mcm = StepModel() + LinearModel()
    
    mc_y = np.nan_to_num(np.append(sliced_vel[1], sliced_vel[2]))
    mc_x = np.nan_to_num(np.append(sliced_vel[9], sliced_vel[9]))

    y_nz = np.where(mc_y != 0.0)
    mc_y = mc_y[y_nz]
    mc_x = mc_x[y_nz]

    mcr = mcm.fit(mc_y, x=mc_x, params=mcp)
    mc_bf = mcr.best_fit

    mc_xrange = np.linspace(-(np.nanmax(x_values)+0.2),(np.nanmax(x_values)+0.2),100)

    ax1.scatter(mc_x, mc_bf, color="#000000", alpha=0.3)
    """

    rgmaps = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
            str(int(cube_id))+"_rotated_maps.npy")

    # --------------------------------------------------#

    # Creating V_OII vs V_* plot
    map_shape = np.shape(rgmaps[0])
    c_x = int(map_shape[0]/2)-1
    c_y = int(map_shape[1]/2)-1

    vd_data = [] # list to store velocities, errors, and radii values

    for cm_y in range(np.shape(rgmaps[0])[1]):
        for cm_x in range(np.shape(rgmaps[0])[0]):
            cpd_stellar = rgmaps[0][cm_y][cm_x] # stellar velocity
            cpd_oii = rgmaps[2][cm_y][cm_x]

            if np.isnan(cpd_stellar) == True:
                pass
            else:
                if (cm_x-c_x)**2/cc_b**2 + (cm_y-c_y)**2/cc_a**2 < 1:
                    cp_sn = rgmaps[4][cm_y][cm_x]                         
                    # top left corner is defined as (x=0,y=0) posn, 
                    # distances between central pixel and curr
                    cpx_dist = cm_x - c_x                 
                    cpy_dist = cm_y - c_y 
                    
                    cp_obs = cpx_dist**2 + cpy_dist**2

                    cp_radius = np.sqrt(cp_obs+cpy_dist**2*np.tan(gal_inc)**2)
                    cp_radius = cp_radius * muse_scale # convert to MUSE scale

                    frac_err_ppxf = a_ppxf # pPXF velocity fractional error
                    rcpd_ppxf = rgmaps[8][cm_y][cm_x]
                    cpds_e = curve(cp_sn, frac_err_ppxf) * rcpd_ppxf # stellar error

                    frac_err_lmfit = a_lmfit # lmfit velocity fractional error
                    rcpd_lmfit = rgmaps[9][cm_y][cm_x]
                    cpdo_e = curve(cp_sn, frac_err_lmfit) * rcpd_lmfit # OII error

                    vd_data.append([cp_radius, cpd_stellar, cpds_e[0], cpd_oii, 
                        cpdo_e[0]])

    vd_data = np.asarray(vd_data) 
    np.save("cube_results/cube_"+str(int(cube_id))+"/cube_"+str(int(cube_id))+
            "_vel_diff_data.npy", vd_data) # save data to an array

    # plotting V_OII-V_* against radius
    vd, vdax1 = plt.subplots(1,1) 
    
    radii = vd_data[:,0]
    v_oii = vd_data[:,3]
    v_star = vd_data[:,1]

    v_diff = (v_oii - v_star) # difference data
    v_diff_err = np.sqrt(vd_data[:,2]**2 + vd_data[:,4]**2) # uncertainty data

    vdax1.errorbar(radii, v_diff, yerr=v_diff_err, ms=5, fmt='o', c="#000000", 
            elinewidth=1.0, capsize=5, capthick=1.0) 

    vdax1.tick_params(labelsize=20)
    vdax1.set_xlabel(r'\textbf{Radius (")}', fontsize=20)
    vdax1.set_ylabel(r'\textbf{V$_{gas}$-V$_{stellar}$ (kms$^{-1}$)}', fontsize=20) 

    vd.tight_layout()
    vd.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_velocity_differences.pdf", bbox_inches="tight")

    # --------------------------------------------------#

    ax1.tick_params(labelsize=20)
    ax1.set_xlabel(r'\textbf{Radius (")}', fontsize=20)
    ax1.set_ylabel(r'\textbf{Velocity (kms$^{-1}$)}', fontsize=20)   
    ax1.legend(loc='lower right', prop={'size': 17})
    ax1.set_xlim([-(np.nanmax(x_values)+0.2), (np.nanmax(x_values)+0.2)])
    g.tight_layout()
    g.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_rotation_curves_1d.pdf", bbox_inches="tight")

    plt.close("all")

    p2ax1.tick_params(labelsize=20)
    p2ax1.set_xlabel(r'\textbf{Radius (")}', fontsize=20)
    p2ax1.set_ylabel(r'\textbf{Velocity (kms$^{-1}$)}', fontsize=20)   
    #p2ax1.legend(loc='lower right', prop={'size': 17})
    #p2ax1.set_xlim([-(np.nanmax(x_values)+0.2), (np.nanmax(x_values)+0.2)])
    p2.tight_layout()
    p2.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_rotation_curves_2d.pdf", bbox_inches="tight")

    plt.close("all")

    # ------------------------------------------------------------------------------ #
    # plotting the offset between gas and stellar velocities in a galaxy
    j, (jax1) = plt.subplots(1,1)

    vel_diff = sliced_vel[2]-sliced_vel[1] # stellar vel - gas vel
    prop_err = np.sqrt(sliced_vel[7]**2 + sliced_vel[8]**2) # propagated error

    jax1.errorbar(sliced_vel[9], vel_diff, yerr=prop_err, ms=5, fmt='o', c="#000000", 
            elinewidth=1.0, capsize=5, capthick=1.0) 

    jax1.set_xlim([-(np.nanmax(sliced_vel[9])+0.2), (np.nanmax(sliced_vel[9])+0.2)])
    jax1.tick_params(labelsize=20)
    jax1.set_xlabel(r'\textbf{Radius (")}', fontsize=20)
    jax1.set_ylabel(r'\textbf{V$_{stellar}$-V$_{gas}$ (kms$^{-1}$)}', fontsize=20) 
    
    j.tight_layout()
    j.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_velocity_offsets.pdf", bbox_inches="tight")
    plt.close("all")

    # ------------------------------------------------------------------------------ #
    # plotting the velocity curves underneath the velocity maps
    h, (hax1, hax2, hax3) = plt.subplots(3, 1, sharex=True, figsize=(4, 8)) 
   
    # Stellar velocity map from pPXF
    hax = hax1.imshow(rotated_galaxy_maps[0], cmap='jet', aspect="auto",
            vmin=ppxf_vel_unique[1], vmax=ppxf_vel_unique[-2]) 
    hax1.set_ylabel(r'\textbf{Stellar}', fontsize=20)

    # Stellar velocity map from lmfit
    hax2.imshow(rotated_galaxy_maps[2], cmap='jet', aspect="auto", 
            vmin=ppxf_vel_unique[1], vmax=ppxf_vel_unique[-2])
    hax2.set_ylabel(r'\textbf{Gas}', fontsize=20)

    # arcseconds labels
    x_scale = sliced_vel[0] - c_x # setting central pixel as radius 0
    x_scale = sliced_vel[0] * muse_scale # converting to MUSE scale
    
    # pPXF stellar velocity 
    hax3.errorbar(sliced_vel[0], sliced_vel[1], yerr=sliced_vel[7], ms=5, c=rot_c[0],
            label=rot_labels[0], elinewidth=1.0, capsize=5, capthick=1.0, fmt='o') 
    # lmfit stellar velocity
    hax3.errorbar(sliced_vel[0], sliced_vel[2], yerr=sliced_vel[8], ms=5, c=rot_c[2],
            label=rot_labels[2], elinewidth=1.0, capsize=5, capthick=1.0, fmt='o') 

    #hax3.set_aspect('auto')

    #hax3.set_xticklabels(sliced_vel[0], x_scale)

    hax3.tick_params(labelsize=20)
    hax3.set_xlabel(r'\textbf{Radius (")}', fontsize=20)
    hax3.set_ylabel(r'\textbf{Velocity (kms$^{-1}$)}', fontsize=20) 
    hax3.legend(loc='lower right', prop={'size': 15})

    #h.colorbar(hax, ax=[hax1, hax2])
    h.tight_layout()
    #h.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+"_velocity.pdf",
            #bbox_inches="tight") 
    plt.close("all")

def rotation_curves_runner():
    # Running to obtain results from pPXF and OII fitting
    cf = ppxf_fitter.cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    uc = ppxf_fitter.usable_cubes(catalogue, bright_objects) # usable cubes
    #uc = uc[3:]
    #uc = np.array([1804])
    print(uc)
    for i_cube in range(len(uc)):
        cube_id = int(uc[i_cube])

        print("cube_"+str(cube_id)+": ")
        
        voronoi_plotter(cube_id)
        rotation_curves(cube_id)

if __name__ == '__main__':
    #voronoi_cube_runner()
    
    #voronoi_runner()
    #voronoi_plotter(849)
    #rotation_curves(849)

    rotation_curves_runner()
