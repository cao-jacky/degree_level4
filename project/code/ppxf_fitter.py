import os

from anybar import AnyBar

import sys
sys.path.insert(0, '/Users/jackycao/Documents/Projects/scripts/')
import personal_scripts

import ppxf_fitter_kinematics_sdss
import ppxf_fitter_gas_population

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import cube_analysis
import ppxf_plots
import cube_reader

import voigt_profiles

import diagnostics

from shutil import copyfile

from astropy.cosmology import FlatLambdaCDM

import spectra_data

import sigma_uncertainties

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def ranged_fitting(cube_id, ranges):
    """ fitting pPXF for different ranges """
    # fitting for the full, original spectrum
    #original = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all")

    #ori_vars = original['variables']
    #ori_errors = original['errors']

    # I want an array to store the parameters that pPXF finds as well
    fit_vars = np.zeros([len(ranges)+1, 4])

    # storing original best parameters and errors
    #fit_vars[0][0] = ori_vars[0]
    #fit_vars[0][1] = ori_vars[1]

    #fit_vars[0][2] = ori_errors[0]
    #fit_vars[0][3] = ori_errors[1]

    for i_range in range(len(ranges)):
        rtc = ranges[i_range]
        print("Considering range " + str(rtc))        
        ranged_fitting = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, rtc)

        rf_vars = ranged_fitting['variables']
        rf_errors = ranged_fitting['errors']
 
        fit_vars[i_range+1][0] = rf_vars[0] # velocity
        fit_vars[i_range+1][1] = rf_vars[1] # sigma velocity dispersion

        fit_vars[i_range+1][2] = rf_errors[0] # velocity error
        fit_vars[i_range+1][3] = rf_errors[1] # sigma velocity dispersion error

    return {'fitted_variables': fit_vars}

def cat_func():
    catalogue = np.load("data/matched_catalogue.npy")
    catalogue = catalogue[catalogue[:,8].argsort()]
    catalogue = catalogue[0:300,:] 
    bright_objects = np.where(catalogue[:,5] < 25.5)[0]
    return {'cat': catalogue, 'bo': bright_objects}

def ignore_cubes():
    cubes_to_ignore = np.array([97,139,140,152,157,159,178,1734,1701,1700,1689,
        1592,1580,1564,1562,1551,1514,1506,1454,1446,1439,1418,1388,1355,1353,1267,
        1227,1210,1198,1187,1159,1157,1137,1132,1121,217,222,317,338,339,343,361,395,
        407,421,438,458,459,481,546,551,582,592,651,659,665,687,710,720,737,744,754,
        789,790,814,834,859,864,878,879,914,965,966,982,1008,1017,1033,1041,1045,
        1063,1068,1114,1162, 112, 722, 764, 769, 760, 1469, 733, 1453, 723, 378,
        135, 474, 1103, 1118, 290, 1181, 1107, 6, 490, 258, 538, 643, 1148, 872,
        1693, 1387, 406, 163, 167, 150, 1320, 1397, 545, 721, 1694, 508, 1311,
        205, 738, 1, 1416, 397, 509, 439, 931, 248, 1402, 1667])
    avoid_objects = np.load("data/avoid_objects.npy")
    avoid_cubes = np.append(cubes_to_ignore, avoid_objects) # create single array

    gas_avoid = np.array([1, 175, 895, 540, 414, 549, 849, 1075])
    
    return {'ac': avoid_cubes,'ga': gas_avoid}

def usable_cubes(catalogue, bright_objects):
    ignoring = ignore_cubes()
    avoid_cubes = ignoring['ac']

    list_usable = []

    for i_cube in range(len(bright_objects)):
        curr_obj = catalogue[i_cube]
        cube_id = int(curr_obj[0])
    
        if (cube_id in avoid_cubes):
            pass
        else:
            list_usable.append(cube_id)

    # testing for individual cubes
    #list_usable = [1804, 1578]
    #list_usable = [1804]

    return list_usable

def ppxf_cube_auto():
    cf = cat_func()
    catalogue = cf['cat'] # calling sorted catalogue from cataogue function
    bright_objects = cf['bo']

    uc = usable_cubes(catalogue, bright_objects) # usable cubes
    print("Loading the following cubes to process:")
    print(uc)
    
    gas_avoid = ignore_cubes()['ga']

    # we want to fit for different regions, providing an array to our region fitting
    # routine of different regions to consider for each cube
    #
    # this array is here as we need to create our data array based off how many 
    # ranges it contains
    ranges = np.array([
        [3700, 4200],
        [3800, 4000],
        [4000, 4500]
        ])
    #ranges = np.array([])

    np.save("data/ppxf_fitting_ranges", ranges)

    # want an array to store various velocity dispersions
    # 1st dimension: an array to store data on every individual cube
    # 2nd dimension: rows of data corresponding to each different range which we are
    #                considering i.e. full spectrum, OII region, absorption regon
    # 3rd dimension: columns storing corresponding data 
    #   [0] : cube_id 
    #   [1] : OII doublet velocity dispersion from lmfit
    #   [2] : sigma for the entire spectrum or regions corresponding to the limits in
    #         ranges from pPXF
    #   [3] : OII doublet velocity dispersion error from lmfit 
    #   [4] : error for sigma from pPXF
    #   [5] : OII doublet velocity dispersion from pPXF
    #   [6] : V-band magnitude from catalogue
    #   [7] : S/N for each cube
    #   [8] : range beginning which is considered
    #   [9] : range end which is considered 
    #   [10] : pPXF sigma from Voigt profile fitter
    #   [11] : sigma from our own Voigt fitter
    #   [12] : fractional error for pPXF for sigma value 
    #   [13] : fractional error for lmfit for sigma value
    #   [14] : velocity value from the pPXF fitting
    #   [15] : error value for the pPXF velocity
    #   [16] : pPXF fractional difference for velocity dispersion
    data = np.zeros([len(uc), 1+len(ranges), 17]) 
 
    for i_cube in range(len(uc)):
        cube_id = int(uc[i_cube])
        print("Currently processing cube " + str(int(cube_id)))

        cube_loc = np.where(catalogue[:,0] == cube_id)[0]
        curr_obj = catalogue[cube_loc][0]
        data[i_cube][0][0] = cube_id

        # Running diagnostics tool for the cube
        #diagnostics.diag_results(cube_id)

        # Processing the kinematic fitting
        variables = ("ppxf_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) + 
                "_variables.npy")
        errors = ("ppxf_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) + 
                "_errors.npy")
        if (os.path.exists(variables) and os.path.exists(errors)):
            # fitting full standard spectrum, and only running if a numpy
            # variables file is not found - saves me the effort of waiting
            kinematic_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id,0, 
                    "all")

            ppxf_vars = kinematic_fit['variables']
            np.save(variables, ppxf_vars)

            ppxf_errors = kinematic_fit['errors']
            np.save(errors, ppxf_errors)
        else:
            ppxf_vars = np.load(variables)
            ppxf_errors = np.load(errors)

        # Processing the Voigt Profiles
        #voigt_sigmas = voigt_profiles.voigt_fitter(cube_id)['sigmas']

        #data[i_cube][0][10] = voigt_sigmas[0]
        #data[i_cube][0][11] = voigt_sigmas[1]

        # Processing the gas fitting to obtain a fitting for the OII doublet
        # fitting for free vs. tied Balmer & [SII]
        """ 
        # turning off the gas fit as there really is no point as it never worked
        # in the first place
        gas_fit = ("ppxf_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) + 
                "_gas_fit.npy")
        if not (os.path.exists(gas_fit)):
            if (cube_id in gas_avoid):
                pass
            else:
                gas_fit1 = ppxf_fitter_gas_population.population_gas_sdss(cube_id, 
                        tie_balmer=False, limit_doublets=False)
                gas_fit2 = ppxf_fitter_gas_population.population_gas_sdss(cube_id, 
                        tie_balmer=True, limit_doublets=True)

                gas_fit1_vars = gas_fit1['variables']
                gas_fit2_vars = gas_fit2['variables']

                gf1_oii = gas_fit1_vars[2][1]
                gf2_oii = gas_fit2_vars[2][1]

                gas_fit_oii = np.average([gf1_oii, gf2_oii])

                gas_fit_vars = np.array([gf1_oii, gf2_oii, gas_fit_oii])
                np.save(gas_fit, gas_fit_vars)
        else:
            if (cube_id in gas_avoid):
                pass
            else:
                gfvs = np.load(gas_fit)
                gas_fit_oii = gfvs[2]

        data[i_cube][0][5] = gas_fit_oii
        """

        # using saved data, rerun analysis to find chi^2s
        ppxf_analysis = ppxf_plots.fitting_plotter(cube_id) 

        ppxf_vel = ppxf_vars[0]
        data[i_cube][0][14] = ppxf_vel
        data[i_cube][0][15] = ppxf_errors[0]

        sigma_stars = ppxf_vars[1]
        data[i_cube][0][2] = sigma_stars

        sigma_stars_error = ppxf_errors[1]
        data[i_cube][0][4] = sigma_stars_error

        # now I want reaccess the lmfit file to obtain sigma and sigma_errors
        cube_result_file = open("cube_results/cube_" + str(cube_id) + "/cube_" + 
                str(cube_id) + "_lmfit.txt")
    
        line_count = 0 
        for crf_line in cube_result_file:
            if (line_count == 19):
                curr_line = crf_line.split()
                sigma_gal = float(curr_line[1])
                sigma_gal_error = float(curr_line[3])
            if (line_count == 20):
                curr_line = crf_line.split()
                z = float(curr_line[1])
            if (line_count == 21):
                curr_line = crf_line.split()
                sigma_inst = float(curr_line[1])
            line_count += 1

        sigma = np.sqrt(sigma_gal**2 + sigma_inst**2)
        sigma_err = np.sqrt((sigma_gal_error/sigma))
        
        # converting sigma into kms^-1 from Å
        c = 299792.458 # speed of light in kms^-1

        vel_dispersion = (sigma / (3727*(1+z))) * c
        data[i_cube][0][1] = vel_dispersion             

        vel_dispersion_err = (sigma_gal_error / (3727*(1+z))) * c
        data[i_cube][0][3] = vel_dispersion_err

        # V-band magnitude (HST 606nm) from catalogue
        data[i_cube][0][6] = curr_obj[5]

        # calculating S/N for each cube
        cube_x_data = np.load("cube_results/cube_" + str(int(cube_id)) + 
                "/cube_" + str(int(cube_id)) + "_cbd_x.npy") 
        cube_y_data = np.load("cube_results/cube_" + str(int(cube_id)) + 
                "/cube_" + str(int(cube_id)) + "_cbs_y.npy")

        sn_region = np.array([4000, 4080]) * (1+z) 
        sn_region_mask = ((cube_x_data > sn_region[0]) & 
                (cube_x_data < sn_region[1]))

        cube_y_sn_region = cube_y_data[sn_region_mask]
        cy_sn_mean = np.mean(cube_y_sn_region)
        cy_sn_std = np.std(cube_y_sn_region)
        cy_sn = cy_sn_mean / cy_sn_std

        data[i_cube][0][7] = cy_sn

        # fractional errors
        frac_err = sigma_uncertainties.uncertainties(cube_id)
        fe_ppxf = frac_err['ppxf']
        fe_lmfit = frac_err['lmfit']

        data[i_cube][0][16] = fe_ppxf  

        uncert_ppxf = fe_ppxf * sigma_stars
        uncert_lmfit = fe_lmfit * vel_dispersion

        # storing as uncertainty values
        data[i_cube][0][12] = uncert_ppxf
        data[i_cube][0][13] = uncert_lmfit

        #print(uncert_ppxf, uncert_lmfit)
        
        # considering different ranges in the spectrum 
        rf = ranged_fitting(cube_id, ranges) # running finder 
        fit_vars = rf['fitted_variables']
        for i_rtc in range(len(ranges)):
            ci = i_rtc+1 # current index in the data array
            # cycling through each individual range and then storing into our
            # data array
            data[i_cube][ci][0] = cube_id

            # velocity dispersion and it's error
            data[i_cube][ci][2] = fit_vars[ci][1]
            data[i_cube][ci][4] = fit_vars[ci][3]

            # ranges used in the fittings
            data[i_cube][ci][8] = ranges[i_rtc][0]
            data[i_cube][ci][9] = ranges[i_rtc][1]     

            # uncertainty for pPXF velocity dispersion from fractional error
            data[i_cube][ci][12] = fe_ppxf * fit_vars[ci][1]

            # velocity value from pPXF
            data[i_cube][ci][14] = fit_vars[ci][0]
            
        # Singular diagnostic plot
        def singular_plot():
            # parameters from lmfit
            lm_params = spectra_data.lmfit_data(cube_id)
            c = lm_params['c']
            i1 = lm_params['i1']
            i2 = lm_params['i2']
            sigma_gal = lm_params['sigma_gal']
            z = lm_params['z']
            sigma_inst = lm_params['sigma_inst']

            f, (ax1, ax2, ax3)  = plt.subplots(1, 3, 
                    gridspec_kw={'width_ratios':[1,1,3]},figsize=(8,2))

            g, (gax1, gax2, gax3) = plt.subplots(1,3,
                    gridspec_kw={'width_ratios':[1,2,2]}, figsize=(12,4))

            #ax1.set_title(r'\textbf{-}')
            ax1.axis('off')
            ax1.text(0.0, 0.9, "cube\_" + str(cube_id), fontsize=13)
            ax1.text(0.0, 0.7, "sigma\_star (km/s): " + 
                    str("{:.1f}".format(sigma_stars)), fontsize=13)
            ax1.text(0.0, 0.55, "sigma\_OII (km/s): " + 
                    str("{:.1f}".format(vel_dispersion)), fontsize=13)

            gax1.axis('off')
            gax1.text(0.0, 0.9, "cube\_" + str(cube_id), fontsize=13)
            gax1.text(0.0, 0.8, "sigma\_star (km/s): " + 
                    str("{:.1f}".format(sigma_stars)), fontsize=13)
            gax1.text(0.0, 0.75, "sigma\_OII (km/s): " + 
                    str("{:.1f}".format(vel_dispersion)), fontsize=13)

            gax1.text(0.0, 0.65, "OII fit outputs: ", fontsize=13)
            gax1.text(0.0, 0.6, "sigma\_gal: " + 
                    str("{:.5f}".format(sigma_gal)), fontsize=13)
            gax1.text(0.0, 0.55, "sigma\_inst: " + 
                    str("{:.5f}".format(sigma_inst)), fontsize=13)


            ax2.set_title(r'\textbf{MUSE}')
            ax2.axis('off') 
            fits_file = ("/Volumes/Jacky_Cao/University/level4/project/" +
                    "cubes_better/cube_" + str(cube_id) + ".fits")
            im_coll_data = cube_reader.image_collapser(fits_file)
            ax2.imshow(im_coll_data['median'], cmap='gray_r')

            # plotting pPXF data
            # defining wavelength as the x-axis
            x_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + 
                    "/cube_" + str(int(cube_id)) + "_lamgal.npy")

            # defining the flux from the data and model
            y_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + 
                    "/cube_" + str(int(cube_id)) + "_flux.npy")
            y_model = np.load("ppxf_results/cube_" + str(int(cube_id)) + 
                    "/cube_" + str(int(cube_id)) + "_model.npy")

            # scaled down y data 
            y_data_scaled = y_data/np.median(y_data) 

            # spectral lines
            sl = spectra_data.spectral_lines() 

            ax3.plot(x_data, y_data_scaled, linewidth=0.7, color="#000000")

            gax2.plot(x_data, y_data_scaled, linewidth=0.7, color="#000000")

            max_y = np.max(y_data_scaled)
            # plotting spectral lines
            for e_key, e_val in sl['emis'].items():
                spec_line = float(e_val)*(1+z)
                spec_label = e_key

                alpha_line = 0.7                            
                alpha_text = 0.75

                ax3.axvline(x=spec_line, linewidth=0.5, color="#1e88e5", 
                        alpha=alpha_line) 
                gax2.axvline(x=spec_line, linewidth=0.5, color="#1e88e5", 
                        alpha=alpha_line)
                gax3.axvline(x=spec_line, linewidth=0.5, color="#1e88e5", 
                        alpha=alpha_line)

            for e_key, e_val in sl['abs'].items():
                spec_line = float(e_val)*(1+z)
                spec_label = e_key

                ax3.axvline(x=spec_line, linewidth=0.5, color="#ff8f00", 
                        alpha=0.7)
                gax2.axvline(x=spec_line, linewidth=0.5, color="#ff8f00", 
                        alpha=0.7)
                gax3.axvline(x=spec_line, linewidth=0.5, color="#1e88e5", 
                        alpha=alpha_line)

            # iron spectral lines
            for e_key, e_val in sl['iron'].items(): 
                spec_line = float(e_val)*(1+z)

                ax3.axvline(x=spec_line, linewidth=0.5, color="#bdbdbd", 
                        alpha=0.3)
                gax2.axvline(x=spec_line, linewidth=0.5, color="#bdbdbd", 
                        alpha=0.3)

            ax3.plot(x_data, y_model, linewidth=0.7, color="#b71c1c")

            ax3.tick_params(labelsize=13)
            ax3.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
            ax3.set_ylabel(r'\textbf{Relative Flux}', fontsize=13)
           
            ax3.set_xlim([3500*(1+z), 4000*(1+z)]) # 3500Å to 4000Å

            gax2.plot(x_data, y_model, linewidth=0.7, color="#b71c1c") 

            xd_range = [np.min(x_data), np.max(x_data)]
            xd = np.linspace(xd_range[0], xd_range[1], 4000)

            # Plotting OII doublet 
            doublet_data = spectra_data.f_doublet(xd*(1+z), c, i1, i2, sigma_gal, z, 
                    sigma_inst)
            ppxf_no_scale = np.load("ppxf_results/cube_" + str(int(cube_id)) + 
                    "/cube_" + str(int(cube_id)) + "_not_scaled.npy")
            gax2.plot(xd, doublet_data/np.median(ppxf_no_scale), lw=0.5, 
                    color="#7b1fa2")

            # Plotting inidividual Gaussians which make the doublet
            dblt_mu = [3727.092, 3729.875] # non-redshifted wavelengths for OII
            l1 = dblt_mu[0] * (1+z)
            l2 = dblt_mu[1] * (1+z)
            
            sig = np.sqrt(sigma_gal**2 + sigma_inst**2) 
            norm = (sig*np.sqrt(2*np.pi))

            x_dat = xd * (1+z)

            lm_y1 = c + ( i1 / norm ) * np.exp(-(x_dat-l1)**2/(2*sig**2))
            lm_y2 = c + ( i2 / norm ) * np.exp(-(x_dat-l2)**2/(2*sig**2))

            gax2.plot(xd, lm_y1/np.median(ppxf_no_scale), linewidth=0.5, 
                    color="#8bc34a", alpha=0.7) 
            gax2.plot(xd, lm_y2/np.median(ppxf_no_scale), linewidth=0.5, 
                    color="#1e88e5", alpha=0.7)

            gax2.tick_params(labelsize=13)
            gax2.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
            gax2.set_ylabel(r'\textbf{Relative Flux}', fontsize=13)
           
            gax2.set_xlim([3700*(1+z), 4000*(1+z)]) # 3700Å to 4000Å

            # Zoomed in plot on the doublet region
            gax3.plot(x_data, y_data_scaled, linewidth=0.7, color="#000000")
            gax3.plot(xd, doublet_data/np.median(ppxf_no_scale), lw=0.5, 
                    color="#7b1fa2")

            gax3.plot(xd, lm_y1/np.median(ppxf_no_scale), linewidth=0.5, 
                    color="#8bc34a", alpha=0.7) 
            gax3.plot(xd, lm_y2/np.median(ppxf_no_scale), linewidth=0.5, 
                    color="#1e88e5", alpha=0.7)

            gax3.tick_params(labelsize=13)
            gax3.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
            gax3.set_ylabel(r'\textbf{Relative Flux}', fontsize=13)
           
            gax3.set_xlim([3700*(1+z), 3750*(1+z)]) # 3700Å to 4000Å

            f.tight_layout()
            f.savefig("diagnostics/single_page/"+str(int(i_cube))+"_cube_"+
                    str(cube_id)+".pdf")

            g.tight_layout()
            g.savefig("diagnostics/single_page_spectra/"+str(int(i_cube))+
                    "_cube_"+str(cube_id)+"_spectra.pdf")

            plt.close("all")

        singular_plot()

    # removing data which doesn't have an O[II] doublet for us to compare to
    sigma_doublet_vals = data[:][:,0][:,1]
    sigma_doublet_zeros = np.where(sigma_doublet_vals == 0)[0]
    data = np.delete(data, sigma_doublet_zeros, 0)
    print(data)

    np.save("data/ppxf_fitter_data", data)

    # tells system to play a sound to alert that work has been finished
    os.system('afplay /System/Library/Sounds/Glass.aiff')
    personal_scripts.notifications("ppxf_fitter", "Script has finished!")

if __name__ == '__main__':
    AnyBar().change('red')

    ppxf_cube_auto()
    #usable_cubes()
    #ppxf_plots.sigma_sn()
    #region_graphs_with_data()

    AnyBar().change('green')
