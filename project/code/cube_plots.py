import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

from mpl_toolkits.axes_grid1 import make_axes_locatable

import multi_cubes
import cube_reader
import ppxf_fitter
import spectra_data

def data_matcher(catalogue_array, cubes_text_file):
    """ matching our cubes file which contains the details of processed cubes """

    catalogue = np.load(catalogue_array)

    cubes_file = open(cubes_text_file)
    cubes_file_num_lines = sum(1 for line in open(str(cubes_text_file))) - 1
    cubes = np.zeros((cubes_file_num_lines, 5))
    file_row_count = 0
    for file_line in cubes_file:
        file_line = file_line.split()
        if (file_row_count == 0):
            pass
        else:
            for file_col in range(len(file_line)):
                cubes[file_row_count-1][file_col] = file_line[file_col]

        file_row_count += 1 

    cubes_file.close()

    # finding which cubes are usable for analysis
    usable_locs = np.where(cubes[:,-1] == 1)[0]
    
    # producing a 'usable data' array where the indices represent the following:
    # 0 = cube_id
    # 1 = catalogue redshift
    # 2 = model redshift
    # 3 = model redshift error
    # 4 = catalogue O[II] flux
    # 5 = model O[II] flux 
    # 6 = model sigma1 
    # 7 = catalogue mag star - I could just calculate from the magnitude?
    usable_data = np.zeros((len(usable_locs), 8))
    for i_cube in range(len(usable_locs)):
        cube_index = usable_locs[i_cube]

        cube_id = int(cubes[cube_index][0])
        usable_data[i_cube][0] = cube_id

        # looking at data from catalogue, column with index 7 has MUSE redshift 
        cat_cube_loc = np.where(catalogue[:,0] == cube_id)[0]
        cat_cube_data = catalogue[cat_cube_loc][0]

        cat_cube_rdst = cat_cube_data[7]
        usable_data[i_cube][1] = cat_cube_rdst

        # looking at our generated model results
        final_results_loc = ("cube_results/cube_" + str(cube_id) + "/cube_" + 
                str(cube_id) + "_lmfit.txt")
        final_results_file = open(final_results_loc) 

        gauss_variables = {'c': 16, 'i1': 15, 'r': 17, 'i2': 18, 'sigma1': 19, 'z': 20}
        gauss_vars = np.zeros((7))
       
        curr_line = 0
        curr_var = 0
        for fr_line in final_results_file:
            if ( 15 <= curr_line <= 21):
                data_line = fr_line.split()
                gauss_vars[curr_var] = data_line[1]
                if (curr_line == 20):
                    rdst_err = data_line[3] 
                curr_var += 1
            curr_line += 1

        rdst_val = gauss_vars[-2]
        usable_data[i_cube][2] = rdst_val
        usable_data[i_cube][3] = rdst_err

        final_results_file.close() 

        # O[II] flux contained in column 12 and 13 depending on OII3726 or OII3729
        cat_cube_flux = np.average([cat_cube_data[12], cat_cube_data[13]])
        usable_data[i_cube][4] = cat_cube_flux

        flux_val = (gauss_vars[1] + gauss_vars[3]) 
        usable_data[i_cube][5] = flux_val

        model_sigma1 = gauss_vars[4]
        usable_data[i_cube][6] = model_sigma1

        # M_star is in column 37
        #cat_mag_star = cat_cube_data[36]
        cat_mag_star = 0
        usable_data[i_cube][7] = cat_mag_star 
    return usable_data

def chisq(model, data, data_err):
    csq = (data-model)**2 / data_err**2
    csq_final = np.sum(csq)

    redcsq = csq_final / (len(csq))
    return {'chisq': csq_final, 'redchisq': redcsq}

def plots(catalogue_array, cubes_text_file):
    data = data_matcher(catalogue_array, cubes_text_file)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    analysis_file = open("results/analysis_results.txt", 'w')

    def redshift():
        rdst_model = data[:,2]
        rdst_cat = data[:,1]
        cube_ids = data[:,0]

        rdst_model_err = data[:,3]

        chi_squared = chisq(rdst_cat, rdst_model, rdst_model_err)
        analysis_file.write("Between the catalogue and model redshift data, we can " +
            "calculate a chi-squared value of " + str(chi_squared['chisq']) + " and "
            + " a reduced chi-squared value of " + str(chi_squared['redchisq']))

        fig, ax = plt.subplots()

        x_min = np.min(rdst_model)
        x_max = np.max(rdst_model)
 
        x = np.linspace(x_min, x_max, 1000) 
        ax.plot(x, x, color="#000000", alpha=0.3, linewidth=1)

        ax.scatter(rdst_model, rdst_cat, color="#000000", s=10)

        for i, txt in enumerate(cube_ids):
            ax.annotate(int(txt), (rdst_model[i], rdst_cat[i]))

        ax.set_title(r'\textbf{Redshift: Catalogue vs. Model}', fontsize=13) 
        ax.set_xlabel(r'\textbf{Model}', fontsize=13)
        ax.set_ylabel(r'\textbf{Catalogue}', fontsize=13)
        fig.savefig("graphs/sanity_checks/redshift.pdf")

    def flux():
        flux_model = data[:,5]
        flux_cat = data[:,4]
        cube_ids = data[:,0]

        fig, ax = plt.subplots()

        x_min = np.min(flux_model)
        x_max = np.max(flux_model)

        ax.scatter(flux_model, flux_cat, color="#000000", s=10)

        for i, txt in enumerate(cube_ids):
            ax.annotate(int(txt), (flux_model[i], flux_cat[i]))

        ax.set_title(r'\textbf{Flux: Catalogue vs. Model}', fontsize=13) 
        ax.set_xlabel(r'\textbf{Model}', fontsize=13)
        ax.set_ylabel(r'\textbf{Catalogue}', fontsize=13)
        fig.savefig("graphs/sanity_checks/flux.pdf")

    def mag_sigma():
        mag_star = data[:,7]
        sigma1 = data[:,6]
        cube_ids = data[:,0]

        fig, ax = plt.subplots()

        ax.scatter(sigma1, mag_star, color="#000000", s=10)

        for i, txt in enumerate(cube_ids):
            ax.annotate(int(txt), (sigma1[i], mag_star[i]))

        ax.set_title(r'\textbf{M}$_{*}$ \textbf{vs.} $\sigma_{1}$', weight='bold', 
                fontsize=13)  
        ax.set_xlabel(r'$\sigma_{1}$', weight='bold', fontsize=13)
        ax.set_ylabel(r'\textbf{M}$_{*}$', weight='bold', fontsize=13)
        fig.savefig("graphs/sanity_checks/mag_sigma.pdf")
   
    redshift()
    flux()
    #mag_sigma()

def seg_overlay(cube_id):
    muse_collapsed = np.load("data/cubes_better/cube_"+str(cube_id)+".npy")
    segmentation = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_segmentation.npy")
    
    fig, ax = plt.subplots()
    ax.imshow(muse_collapsed, cmap='gray_r')
    ax.imshow(segmentation, cmap="Blues", alpha=0.5)
    
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_seg_overlayed.pdf")
    plt.close("all")

def spectra(cube_id):
    # parameters from lmfit
    lm_params = spectra_data.lmfit_data(cube_id)
    z = lm_params['z']

    # defining wavelength as the x-axis
    x_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_lamgal.npy")

    # defining the flux from the data and model
    y_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_flux.npy")
    y_model = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_model.npy")

    # scaling y data using the median of the data
    y_data_scaled = y_data/np.median(y_data)
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x_data, y_data_scaled, linewidth=2, color="#000000")
    ax.plot(x_data, y_model, linewidth=2, color="#b71c1c")

    # residuals
    residual = y_data_scaled - y_model
    res_median = np.median(residual)
    res_stddev = np.std(residual)    
    mask = ((residual < res_stddev) & (residual > -res_stddev))     

    ax.scatter(x_data[mask], residual[mask]-1, s=10, color="#43a047", alpha=0.5)

    # spectral lines
    sl = spectra_data.spectral_lines()
    doublets = np.array([3727.092, 3728.875]) * (1+z)
    max_y = np.max(y_data_scaled)

    for e_key, e_val in sl['emis'].items():
        spec_line = float(e_val)*(1+z)
        spec_label = e_key

        if (e_val in str(doublets)):
            alpha_line = 0.2
        else:
            alpha_line = 0.7
            
        alpha_text = 0.75

        ax.axvline(x=spec_line, linewidth=1.5, color="#1e88e5", alpha=alpha_line)
        ax.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=alpha_text,
                weight="bold", fontsize=15) 

    for e_key, e_val in sl['abs'].items():
        spec_line = float(e_val)*(1+z)
        spec_label = e_key

        ax.axvline(x=spec_line, linewidth=1.5, color="#ff8f00", alpha=0.7)
        ax.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=0.75,
                weight="bold", fontsize=15)

    # iron spectral lines
    for e_key, e_val in sl['iron'].items(): 
        spec_line = float(e_val)*(1+z)

        ax.axvline(x=spec_line, linewidth=0.5, color="#bdbdbd", alpha=0.3)

    ax.tick_params(labelsize=33)
    ax.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=20)
    ax.set_ylabel(r'\textbf{Relative flux}', fontsize=20)

    plt.tight_layout()
    plt.savefig("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_spectra_complete.pdf", bbox_inches="tight")
    plt.close("all")

def cube_table():
    pass

def auto_runner():
    # Running through the usable sample
    cf = ppxf_fitter.cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    uc = ppxf_fitter.usable_cubes(catalogue, bright_objects) # usable cubes
    #uc = np.array([1804, 1578])
    uc = uc[0:6] # 0:11 is the full usable sample for Voronoi plots
    print(uc)

    fig, axs = plt.subplots(len(uc), 9, figsize=(64, 38), gridspec_kw={'hspace':0.3,
        'wspace':0.45, 'width_ratios':[7,7,15,7,7,7,10,10,10]})

    for i_cube in range(len(uc)):
        cube_id = int(uc[i_cube])

        print("cube_"+str(cube_id))

        seg_overlay(cube_id) # creating image of galaxy with segmentation map overlayed
        spectra(cube_id) # creating a spectra

        # changing backgrounds of maps from white to black
        #current_cmap = plt.cm.jet
        #current_cmap.set_bad(color='white')

        # Remaking old plots
        #cube_reader.analysis("/Volumes/Jacky_Cao/University/level4/project/"+
                #"cubes_better/cube_"+str(cube_id)+".fits", 
                #"data/skyvariance_csub.fits")

        # HST colour image
        hst_colour = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
                +"_coloured_image_data.npy")

        axs[i_cube,0].imshow(hst_colour, interpolation='nearest', aspect="auto")
        axs[i_cube,0].set_axis_off()

        hsts = np.shape(hst_colour) # shape of the HST frame

        # cube_id label 
        axs[i_cube,0].text(hsts[0]*0.5, hsts[1]*0.95, r'\textbf{C'+str(cube_id)+'}', 
                {'color': "#ffffff", 'fontsize': 40}, horizontalalignment='center',
                weight='heavy')

        # --------------------------------------------------#
       
        # MUSE collapsed image and segmentation map
        muse_collapsed = np.load("data/cubes_better/cube_"+str(cube_id)+".npy")
        segmentation = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
            "_segmentation.npy")
    
        axs[i_cube,1].imshow(muse_collapsed, cmap='gray_r', aspect="auto")
        axs[i_cube,1].imshow(segmentation, cmap="Blues", alpha=0.5, aspect="auto")
        #axs[i_cube,1].set_axis_off()
        axs[i_cube,1].tick_params(labelsize=33)
        axs[i_cube,1].set_ylabel(r'\textbf{Pixels}', fontsize=40)

        # --------------------------------------------------#

        # Spectra
        # parameters from lmfit
        lm_params = spectra_data.lmfit_data(cube_id)
        c = lm_params['c']
        i1 = lm_params['i1']
        i2 = lm_params['i2']
        sigma_gal = lm_params['sigma_gal']
        z = lm_params['z']
        sigma_inst = lm_params['sigma_inst']

        # spectral lines
        sl = spectra_data.spectral_lines() 

        doublets = np.array([3727.092, 3728.875]) * (1+z)

        # plotting spectral lines
        for e_key, e_val in sl['emis'].items():
            spec_line = float(e_val)*(1+z)
            spec_label = e_key

            if (e_val in str(doublets)):
                alpha_line = 0.2
            else:
                alpha_line = 0.7

            axs[i_cube,2].axvline(x=spec_line, linewidth=2, color="#1e88e5", 
                    alpha=alpha_line)

        for e_key, e_val in sl['abs'].items():
            spec_line = float(e_val)*(1+z)
            spec_label = e_key

            axs[i_cube,2].axvline(x=spec_line, linewidth=2, color="#ff8f00", alpha=0.7)

        # iron spectral lines
        for e_key, e_val in sl['iron'].items(): 
            spec_line = float(e_val)*(1+z)

            axs[i_cube,2].axvline(x=spec_line, linewidth=2, color="#bdbdbd", alpha=0.3)

        # defining wavelength as the x-axis
        x_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_lamgal.npy")

        # defining the flux from the data and model
        y_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_flux.npy")
        y_model = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_model.npy")

        # scaling y data using the median of the data
        y_data_scaled = y_data/np.median(y_data)
        
        axs[i_cube,2].plot(x_data, y_data_scaled, linewidth=2, color="#000000")
        
        # plotting over the OII doublet 
        dblt_av = np.average(doublets)

        dblt_x_mask = ((x_data > dblt_av-20) & (x_data < dblt_av+20))
        doublet_x_data = x_data[dblt_x_mask]
        doublet_data = spectra_data.f_doublet(doublet_x_data, c, i1, i2, sigma_gal, z, 
                sigma_inst)
        doublet_data = doublet_data / np.median(y_data)
        axs[i_cube,2].plot(doublet_x_data, doublet_data, linewidth=2, 
                color="#9c27b0")

        axs[i_cube,2].plot(x_data, y_model, linewidth=2, color="#b71c1c")

        axs[i_cube,2].tick_params(labelsize=33) 
        axs[i_cube,2].set_ylabel(r'\textbf{Relative flux}', fontsize=40)
        
        # --------------------------------------------------#

        # Voronoi data
        voronoi_map = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
                str(int(cube_id))+"_voronoi_map.npy")
        axs[i_cube,3].imshow(voronoi_map, cmap="prism", aspect="auto")
        axs[i_cube,3].imshow(segmentation, cmap="Blues", alpha=0.5, aspect="auto")
        axs[i_cube,3].tick_params(labelsize=33)
        axs[i_cube,3].set_ylabel(r'\textbf{Pixels}', fontsize=40)

        # Galaxy maps
        galaxy_maps = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
            str(int(cube_id))+"_maps.npy")

        # Rotated maps 
        rotated_maps = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
            str(int(cube_id))+"_rotated_maps.npy")

        # finding central pixel
        map_shape = np.shape(rotated_maps[0])
        c_x = int(map_shape[0]/2)-1
        c_y = int(map_shape[1]/2)-1

        # scaling by pPXF maps
        ppxf_vel_data = rotated_maps[0]
        ppxf_vel_unique = np.unique(ppxf_vel_data)

        # V_OII map (lmfit)
        voronoi_map = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
                str(int(cube_id))+"_voronoi_map.npy")
        voii = axs[i_cube,4].imshow(rotated_maps[2], cmap="jet", aspect="auto",
                vmin=np.nanmin(ppxf_vel_unique), vmax=np.nanmax(ppxf_vel_unique))
        axs[i_cube,4].tick_params(labelsize=33)
        
        # amount to shift the colour bar with respect to the y-axis
        if i_cube == 2:
            height_amount = 1.8  
        if i_cube == 3:
            height_amount = 2.3
        if i_cube == 4:
            height_amount = 0.0005
        if i_cube == 5:
            height_amount = 4.95
        else:
            height_amount = i_cube

        cb_ax = fig.add_axes([0.515, 1-(0.195+0.136*height_amount), 0.015, 0.007])
        fcbar = plt.colorbar(voii, ax=axs[i_cube,4], cax=cb_ax, 
                orientation='horizontal')

        fcbar.ax.tick_params(labelsize=20, rotation=90)
        fcbar.ax.set_title(r'\textbf{(kms$^{-1}$)}', fontsize=20, pad=7)

        # V_star map (pPXF)
        voronoi_map = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
                str(int(cube_id))+"_voronoi_map.npy")
        vstar = axs[i_cube,5].imshow(rotated_maps[0], cmap="jet", aspect="auto",
                vmin=np.nanmin(ppxf_vel_unique), vmax=np.nanmax(ppxf_vel_unique))
        axs[i_cube,5].tick_params(labelsize=33)

        cb_ax = fig.add_axes([0.5915, 1-(0.195+0.136*height_amount), 0.015, 0.007])
        fcbar = plt.colorbar(vstar, ax=axs[i_cube,5], cax=cb_ax, 
                orientation='horizontal')

        fcbar.ax.tick_params(labelsize=20, rotation=90)
        fcbar.ax.set_title(r'\textbf{(kms$^{-1}$)}', fontsize=20, pad=7) 

        # --------------------------------------------------#

        # 1D rotation curve 

        muse_scale = 0.20 # MUSE pixel scale in arcsec/pixel

        # slice containing the Voronoi IDs
        vid_slice = np.nanmedian(rotated_maps[7][c_y-1:c_y+2,:], axis=0)
        vid_slice = np.nan_to_num(vid_slice)

        # unique Voronoi IDs and their locations 
        unique_vids, unique_locs = np.unique(vid_slice.astype(int), return_index=True)

        # select out a horizontal strip based on central pixel
        ppxf_map_slice = rotated_maps[0][c_y-1:c_y+2,:]
        ppxf_map_median = np.nanmedian(ppxf_map_slice, axis=0)
        ppxf_map_median = ppxf_map_median[unique_locs] # masking out repeated values

        lmfit_map_slice = rotated_maps[2][c_y-1:c_y+2,:]
        lmfit_map_median = np.nanmedian(lmfit_map_slice, axis=0)
        lmfit_map_median = lmfit_map_median[unique_locs] # masking out repeated values

        # loading "a" factors in a/x model
        a_ppxf = np.load("uncert_ppxf/vel_curve_best_values_ppxf.npy")
        a_lmfit = np.load("uncert_lmfit/vel_curve_best_values_lmfit.npy")

        sn_slice = np.nanmedian(rotated_maps[4][c_y-1:c_y+2,:], axis=0)[unique_locs]
       
        # using full velocity maps (not de-redshifted ones)
        ppxf_map_fv_slice = rotated_maps[8][c_y-1:c_y+2,:]
        ppxf_map_fv_median = np.nanmedian(ppxf_map_fv_slice, axis=0)
        ppxf_map_fv_median = ppxf_map_fv_median[unique_locs] 

        lmfit_map_fv_slice = rotated_maps[9][c_y-1:c_y+2,:]
        lmfit_map_fv_median = np.nanmedian(lmfit_map_fv_slice, axis=0)
        lmfit_map_fv_median = lmfit_map_fv_median[unique_locs] 

        # pPXF velocity fractional error
        frac_err_ppxf = (a_ppxf/sn_slice) * ppxf_map_fv_median
        ppxf_y_err = frac_err_ppxf

        # lmfit velocity fractional error
        frac_err_lmfit = (a_lmfit/sn_slice) * lmfit_map_fv_median
        lmfit_y_err = frac_err_lmfit

        # array which defines the x-scale 
        x_scale = np.arange(0, map_shape[0], 1.0)  

        x_scale = x_scale - c_x # setting central pixel as radius 0
        x_scale = x_scale * muse_scale # converting to MUSE scale
        x_values = x_scale[unique_locs] # masking out repeated values

        axs[i_cube,6].errorbar(x_values, ppxf_map_median, yerr=ppxf_y_err, 
                ms=10, fmt='o', c='#03a9f4', elinewidth=2, capsize=10, capthick=2,
                label=r'\textbf{Stars}')
        axs[i_cube,6].errorbar(x_values, lmfit_map_median, yerr=lmfit_y_err, 
                ms=10, fmt='o', c='#f44336', elinewidth=2, capsize=10, capthick=2,
                label=r'\textbf{Gas}')

        axs[i_cube,6].tick_params(labelsize=33)
        axs[i_cube,6].set_ylabel(r'\textbf{Velocity (kms$^{-1}$)}', fontsize=40)

        lgnd = axs[i_cube,6].legend(loc='upper left', prop={'size': 20})
        lgnd.get_frame().set_alpha(0.5)

        for tick in axs[i_cube,6].get_yticklabels():
            # rotating y-axis labels 
            tick.set_rotation(90)
        
        # --------------------------------------------------#

        # 2D rotation curve
        twod_data = np.load("cube_results/cube_"+str(int(cube_id))+"/cube_"+
                str(int(cube_id))+"_vel_diff_data.npy")

        radii = twod_data[:,0]
        v_oii = twod_data[:,3]
        v_star = twod_data[:,1]

        axs[i_cube,7].errorbar(radii, np.abs(v_star), yerr=twod_data[:,2], 
                ms=10, fmt='o', c='#03a9f4', elinewidth=2, capsize=10, capthick=2,
                label=r'\textbf{Stars}')
        axs[i_cube,7].errorbar(radii, np.abs(v_oii), yerr=twod_data[:,4], 
                ms=10, fmt='o', c='#f44336', elinewidth=2, capsize=10, capthick=2,
                label=r'\textbf{Gas}')

        axs[i_cube,7].tick_params(labelsize=33)

        lgnd = axs[i_cube,7].legend(loc='upper left', prop={'size': 20})
        lgnd.get_frame().set_alpha(0.5)

        # --------------------------------------------------#

        # overlaying sliced regions for gas and stellar
        overlayed_slice = rotated_maps[2]
        overlayed_slice[np.where(overlayed_slice != 1.0)] = np.nan
        overlayed_slice[c_y-1:c_y+2,:] = 2.0

        axs[i_cube,4].imshow(overlayed_slice, cmap='gray', alpha=0.5, aspect="auto")
    
        overlayed_slice = rotated_maps[0]
        overlayed_slice[np.where(overlayed_slice != 1.0)] = np.nan
        overlayed_slice[c_y-1:c_y+2,:] = 2.0

        axs[i_cube,5].imshow(overlayed_slice, cmap='gray', alpha=0.5, aspect="auto")

        # --------------------------------------------------#

        # V_OII-V_* vs. radius plots
        v_diff = (v_oii - v_star) # difference data
        v_diff_err = np.sqrt(twod_data[:,2]**2 + twod_data[:,4]**2) # uncertainty data

        axs[i_cube,8].errorbar(radii, v_diff, yerr=v_diff_err, ms=10, fmt='o', 
                c="#000000", elinewidth=2, capsize=10, capthick=2)

        axs[i_cube,8].tick_params(labelsize=33)  
        axs[i_cube,8].set_ylabel(r'\textbf{V$_{OII}$-V$_{*}$ (kms$^{-1}$)}', 
                fontsize=40)

        for tick in axs[i_cube,8].get_yticklabels():
            # rotating y-axis labels 
            tick.set_rotation(90)

        # --------------------------------------------------#

        # Add title for only first row of plots
        if cube_id == uc[0]:
            axs[i_cube,0].set_title(r'\textbf{HST}', fontsize=40, pad=18)
            axs[i_cube,1].set_title(r'\textbf{MUSE}', fontsize=40, pad=18)
            axs[i_cube,2].set_title(r'\textbf{Spectra}', fontsize=40, pad=18)
            axs[i_cube,3].set_title(r'\textbf{Voronoi map}', fontsize=40, pad=18)
            axs[i_cube,4].set_title(r'\textbf{V$_{OII}$ map}', fontsize=40, pad=18)
            axs[i_cube,5].set_title(r'\textbf{V$_{*}$ map}', fontsize=40, pad=18)
            axs[i_cube,6].set_title(r'\textbf{1D Rot. Curve}', fontsize=40, pad=18)
            axs[i_cube,7].set_title(r'\textbf{2D Rot. Curve}', fontsize=40, pad=18)
            axs[i_cube,8].set_title(r'\textbf{V$_{OII}$-V$_{*}$}', 
                    fontsize=40, pad=18)

        # Add labels for only edge plots
        if cube_id == uc[-1]:
            axs[i_cube,1].set_xlabel(r'\textbf{Pixels}', fontsize=40)
            
            axs[i_cube,2].set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=40)

            axs[i_cube,3].set_xlabel(r'\textbf{Pixels}', fontsize=40)
            axs[i_cube,4].set_xlabel(r'\textbf{Pixels}', fontsize=40)
            axs[i_cube,5].set_xlabel(r'\textbf{Pixels}', fontsize=40)
        
            axs[i_cube,6].set_xlabel(r'\textbf{Radius (")}', fontsize=40)
            axs[i_cube,7].set_xlabel(r'\textbf{Radius (")}', fontsize=40) 
            axs[i_cube,8].set_xlabel(r'\textbf{Radius (")}', fontsize=40)

    #fig.tight_layout()
    fig.savefig("graphs/spectra_complete.pdf", bbox_inches="tight")
    

if __name__ == '__main__':
    file_catalogue  = "data/matched_catalogue.npy"
    file_cubes      = "data/cubes.txt"
    #plots(file_catalogue, file_cubes)

    auto_runner()
