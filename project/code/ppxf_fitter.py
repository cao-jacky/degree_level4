import os

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

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def ppxf_cubes(cube_id):
    print("")
    print("Currently processing on cube " + str(int(cube_id)))
    print("Processing kinematic fitting: ")
    kin_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all") 

    # don't need to consider mass fittings, just kinematic ones
    """
    print("Processing mass fitting with free Balmer & [SII]: ")
    ppxf_fitter_gas_population.population_gas_sdss(cube_id, tie_balmer=False, limit_doublets=False)
    print("")
    print("Processing mass fitting with tied Balmer & [SII]: ")
    ppxf_fitter_gas_population.population_gas_sdss(cube_id, tie_balmer=True, limit_doublets=True)
    print("\n")
    """

    return {'kinematic_fitting': kin_fit}

def ranged_fitting(cube_id, ranges):
    """ fitting pPXF for different ranges """
    # fitting for the full, original spectrum
    original = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all")

    ori_vars = original['variables']
    ori_errors = original['errors']

    # I want an array to store the parameters that pPXF finds as well
    fit_vars = np.zeros([len(ranges)+1, 4])

    # storing original best parameters and errors
    fit_vars[0][0] = ori_vars[0]
    fit_vars[0][1] = ori_vars[1]

    fit_vars[0][2] = ori_errors[0]
    fit_vars[0][3] = ori_errors[1]

    for i_range in range(len(ranges)):
        rtc = ranges[i_range]
        ranged_fitting = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, rtc)

        rf_vars = ranged_fitting['variables']
        rf_errors = ranged_fitting['errors']

        fit_vars[i_range+1][0] = rf_vars[0] # sigma velocity
        fit_vars[i_range+1][1] = rf_vars[1] # sigma velocity dispersion

        fit_vars[i_range+1][2] = rf_errors[0] # sigma velocity error
        fit_vars[i_range+1][3] = rf_errors[1] # sigma velocity dispersion error

    return {'fitted_variables': fit_vars}

def region_graphs_with_data():
    data = np.load("data/ppxf_fitter_data.npy")
    # graphs to consider for regions: lam_1 vs lam_2, lam_2 vs lam_3, 
    # lam_3 vs lam_4, lam_4 vs lam_5

    for i_range in range(4):
        fig, ax = plt.subplots()
        for i in range(len(data[:][:,0])):
            sigma_x = data[:][i,i_range+1][2]
            sigma_y = data[:][i,i_range+2][2]

            ax.scatter(sigma_x, sigma_y, color="#000000", s=10)

            curr_id = data[:][i,0][0]
            
            ax.annotate(int(curr_id), (sigma_x, sigma_y))

        ax.tick_params(labelsize=15)
        ax.set_xlabel(r'\textbf{$\sigma_{'+str(i_range+1)+'}$}', fontsize=15)
        ax.set_ylabel(r'\textbf{$\sigma_{'+str(i_range+2)+'}$}', fontsize=15)

        fig.tight_layout()
        fig.savefig("graphs/regions/sigma_ranges_"+str(i_range)+".pdf")
        plt.close("all") 

def ppxf_cube_auto():
    catalogue = np.load("data/matched_catalogue.npy")
    catalogue = catalogue[catalogue[:,8].argsort()]
    catalogue = catalogue[0:300,:] 

    bright_objects = np.where(catalogue[:,5] < 25.0)[0]

    avoid_objects = np.load("data/avoid_objects.npy")
    cubes_to_ignore = np.array([97,139,140,152,157,159,178,1734,1701,1700,1689,
        1592,1580,1564,1562,1551,1514,1506,1454,1446,1439,1418,1388,1355,1353,1267,
        1227,1210,1198,1187,1159,1157,1137,1132,1121,217,222,317,338,339,343,361,395,
        407,421,438,458,459,481,546,551,582,592,651,659,665,687,710,720,737,744,754,
        789,790,814,834,859,864,878,879,914,965,966,982,1008,1017,1033,1041,1045,
        1063,1068,1114,1162, 112, 722, 764, 769, 760, 1469, 733, 1453, 723, 378,
        135, 474, 1103, 1118, 290, 1181, 1107, 6, 490, 258, 538, 643, 1148, 872,
        1693, 1387, 406, 163, 167, 150, 1320, 1397, 545, 721, 1694, 508, 1311,
        205, 738])

    gas_avoid = np.array([1, 175, 895, 540, 414, 549, 849, 1075]) 

    # we want to fit for different regions, providing an array to our region fitting
    # routine of different regions to consider for each cube
    #
    # this array is here as we need to create our data array based off how many 
    # ranges it contains
    ranges = np.array([
        [3540, 3860],
        [3860, 4180],
        [4180, 4500],
        [3500, 3750],
        [3750, 4500]
        ])

    # want an array to store various velocity dispersions
    # 1st dimension: an array to store data on every individual cube
    # 2nd dimension: rows of data corresponding to each different range which we are
    #                considering i.e. full spectrum, OII region, absorption regon
    # 3rd dimension: columns storing corresponding data 
    #   [0] : cube_id 
    #   [1] : OII doublet velocity dispersion 
    #   [2] : sigma for the entire spectrum or regions corresponding to the limits in
    #         ranges
    #   [3] : OII doublet velocity dispersion error
    #   [4] : error for sigma from pPXF
    #   [5] : OII doublet velocity dispersion from pPXF
    #   [6] : V-band magnitude from catalogue
    #   [7] : S/N for each cube
    #   [8] : range beginning which is considered
    #   [9] : range end which is considered 
    #   [10] : pPXF sigma from Voigt profile fitter
    #   [11] : sigma from our own Voigt fitter
    data = np.zeros([len(bright_objects), 1+len(ranges), 12])

    # testing code for just pPXF
    #bright_objects = np.array([0])
    #cloc = np.where([catalogue[:,0] == 1804])[1]
    #catalogue = catalogue[cloc]

    cube_counter = 0

    with PdfPages('diagnostics/single_page/mag_ranked.pdf') as pdf: 
        for i_cube in range(len(bright_objects)):
            curr_obj = catalogue[i_cube]
            cube_id = int(curr_obj[0])

            data[i_cube][0][0] = cube_id

            if (cube_counter < 10):
                if ((cube_id in avoid_objects) or (cube_id in cubes_to_ignore)):
                    pass
                else:
                    print("Currently processing cube " + str(int(cube_id)))

                    # Running diagnostics tool for the cube
                    diagnostics.diag_results(cube_id)

                    cube_file = ("/Volumes/Jacky_Cao/University/level4/project/" + 
                            "cubes_better/" + "cube_" + str(cube_id) + ".fits")
                    desktop_folder_file = ("/Users/jackycao/Desktop/fits_files/" + 
                            "cube_" + str(cube_id) + ".fits") 
                    #copyfile(cube_file, desktop_folder_file)

                    # Processing the kinematic fitting
                    variables = ("ppxf_results/cube_" + str(cube_id) + "/cube_" + 
                            str(cube_id) + "_variables.npy")
                    errors = ("ppxf_results/cube_" + str(cube_id) + "/cube_" + 
                            str(cube_id) + "_errors.npy")
                    if not (os.path.exists(variables) and os.path.exists(errors)):
                        # fitting full standard spectrum, and only running if a numpy
                        # variables file is not found - saves me the effort of waiting
                        kinematic_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(
                                cube_id,0 , "all")

                        ppxf_vars = kinematic_fit['variables']
                        np.save(variables, ppxf_vars)

                        ppxf_errors = kinematic_fit['errors']
                        np.save(errors, ppxf_errors)
                    else:
                        ppxf_vars = np.load(variables)
                        ppxf_errors = np.load(errors)

                    # Processing the Voigt Profiles
                    voigt_sigmas = voigt_profiles.voigt_fitter(cube_id)['sigmas']

                    data[i_cube][0][10] = voigt_sigmas[0]
                    data[i_cube][0][11] = voigt_sigmas[1]

                    # Processing the gas fitting to obtain a fitting for the OII doublet
                    # fitting for free vs. tied Balmer & [SII]
                    gas_fit = ("ppxf_results/cube_" + str(cube_id) + "/cube_" + 
                            str(cube_id) + "_gas_fit.npy")

                    if not (os.path.exists(gas_fit)):
                        if (cube_id in gas_avoid):
                            pass
                        else:
                            gas_fit1 = ppxf_fitter_gas_population.population_gas_sdss(
                                    cube_id, tie_balmer=False, limit_doublets=False)
                            gas_fit2 = ppxf_fitter_gas_population.population_gas_sdss(
                                    cube_id, tie_balmer=True, limit_doublets=True)

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

                    # using saved data, rerun analysis to find chi^2s
                    ppxf_analysis = ppxf_plots.fitting_plotter(cube_id) 

                    sigma_stars = ppxf_vars[1]
                    data[i_cube][0][2] = sigma_stars

                    sigma_stars_error = ppxf_errors[1]
                    data[i_cube][0][4] = sigma_stars_error

                    # now I want reaccess the lmfit file to find OII sigma
                    cube_result_file = open("cube_results/cube_" + str(cube_id) + 
                            "/cube_" + str(cube_id) + "_lmfit.txt")
                
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
                    
                    # converting sigma into kms^-1 units (?): sigma is in wavelength units
                    # therefore apply lambda*v=c, therefore v = c/lambda
                    c = 299792.458 # speed of light in kms^-1
                    v = c / (sigma * 10**(3)) # velocity dispersion as a velocity

                    # the 'sigma' aka velocity dispersion and the associated error
                    vel_dispersion = c / (sigma * 10**(3))
            
                    data[i_cube][0][1] = vel_dispersion              
                    data[i_cube][0][3] = c / (sigma_gal_error * 10**(3))   

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
         
                    # considering different ranges in the spectrum 
                    """
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
                        data[i_cube][ci][9] = ranges[i_rtc][1]"""       

                    # Singular diagnostic plot

                    def singular_plot():
                        f, (ax1, ax2, ax3)  = plt.subplots(1, 3, 
                                gridspec_kw={'width_ratios':[1,1,3]},figsize=(8,2))

                        # "{:.1f}".format(voigt_sigmas[0])

                        #ax1.set_title(r'\textbf{-}')
                        ax1.axis('off')
                        ax1.text(0.0, 0.9, "cube\_" + str(cube_id), fontsize=13)
                        ax1.text(0.0, 0.7, "sigma\_star: " + 
                                str("{:.1f}".format(voigt_sigmas[0])), fontsize=13)
                        ax1.text(0.0, 0.55, "sigma\_OII: " + 
                                str("{:.1f}".format(vel_dispersion)), fontsize=13)

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
                        sl = {
                                'emis': {
                                    '':             '3727.092', 
                                    'OII':          '3728.875',
                                    'HeI':          '3889.0',
                                    'SII':          '4072.3',
                                    'H$\delta$':    '4101.89',
                                    'H$\gamma$':    '4341.68'
                                    },
                                'abs': {
                                    r'H$\theta$':   '3798.976',
                                    'H$\eta$':      '3836.47',
                                    'CaK':          '3934.777',
                                    'CaH':          '3969.588',
                                    'G':            '4305.61' 
                                    },
                                'iron': {
                                    'FeI1':     '4132.0581',
                                    'FeI2':     '4143.8682',
                                    'FeI3':     '4202.0293', 
                                    'FeI4':     '4216.1836',
                                    'FeI5':     '4250.7871',
                                    'FeI6':     '4260.4746',
                                    'FeI7':     '4271.7607',
                                    'FeI8':     '4282.4028',
                                    }
                                }


                        plt.figure()

                        ax3.plot(x_data, y_data_scaled, linewidth=1.1, color="#000000")

                        #ax3.set_title(r'\textbf{cube\_'+str(cube_id)+'}') 

                        max_y = np.max(y_data_scaled)
                        # plotting spectral lines
                        for e_key, e_val in sl['emis'].items():
                            spec_line = float(e_val)
                            #spec_line = float(e_val) * (1+z)
                            spec_label = e_key

                            alpha_line = 0.7                            
                            alpha_text = 0.75

                            ax3.axvline(x=spec_line, linewidth=0.5, color="#1e88e5", 
                                    alpha=alpha_line) 

                        for e_key, e_val in sl['abs'].items():
                            spec_line = float(e_val)
                            #spec_line = float(e_val) * (1+z)
                            spec_label = e_key

                            ax3.axvline(x=spec_line, linewidth=0.5, color="#ff8f00", 
                                    alpha=0.7)

                        # iron spectral lines
                        for e_key, e_val in sl['iron'].items(): 
                            spec_line = float(e_val)
                            #spec_line = float(e_val) * (1+z)

                            ax3.axvline(x=spec_line, linewidth=0.5, color="#bdbdbd", 
                                    alpha=0.3)

                        ax3.plot(x_data, y_model, linewidth=1.5, color="#b71c1c")

                        ax3.tick_params(labelsize=13)
                        ax3.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
                        ax3.set_ylabel(r'\textbf{Relative Flux}', fontsize=13)
                        
                        f.tight_layout()
                        pdf.savefig()
                        f.savefig("diagnostics/single_page/cube_"+str(cube_id)+".pdf")
                        plt.close() 

                    singular_plot()
                    cube_counter += 1
            else:
                pass

    # removing data which doesn't have an O[II] doublet for us to compare to
    sigma_doublet_vals = data[:][:,0][:,1]
    sigma_doublet_zeros = np.where(sigma_doublet_vals == 0)[0]
    data = np.delete(data, sigma_doublet_zeros, 0)
    print(data)

    np.save("data/ppxf_fitter_data", data)

    # working with just one cube
    #cube_id = 1129
    #ppxf_cubes(cube_id)
    #ranged_fitting(cube_id)
    #ppxf_plots.fitting_plotter(cube_id)

    # we want to plot sigma_stars vs. to sigma_OII
    def sigma_stars_vs_sigma_oii():
        fig, ax = plt.subplots()

        # yerr=data[:][:,0][:,4]
        ax.errorbar(data[:][:,0][:,1], data[:][:,0][:,2], xerr=data[:][:,0][:,3], 
                color="#000000", fmt="o", elinewidth=1.0, 
                capsize=5, capthick=1.0)

        for i in range(len(data[:][:,0])):
            curr_id = data[:][i,0][0]
            curr_x = data[:][i,0][1]
            curr_y = data[:][i,0][2]

            ax.annotate(int(curr_id), (curr_x, curr_y))

        ax.tick_params(labelsize=15)
        ax.set_ylabel(r'\textbf{$\sigma_{*}$}', fontsize=15)
        ax.set_xlabel(r'\textbf{$\sigma_{OII}$}', fontsize=15)

        fig.tight_layout()
        fig.savefig("graphs/sigma_star_vs_sigma_oii.pdf")
        plt.close("all") 

    def oii_lmfit_vs_oii_ppxf():
        fig, ax = plt.subplots()

        for i in range(len(data[:][:,0])):
            curr_id = data[:][i,0][0]
            if (curr_id in gas_avoid):
                pass 
            else:
                curr_x = data[:][i,0][1]
                curr_y = data[:][i,0][5]
                ax.scatter(curr_x, curr_y, color="#000000", s=10)

                ax.annotate(int(curr_id), (curr_x, curr_y))

        ax.tick_params(labelsize=15)
        ax.set_xlabel(r'\textbf{$\sigma_{OII_{lmfit}}$}', fontsize=15)
        ax.set_ylabel(r'\textbf{$\sigma_{OII_{pPXF}}$}', fontsize=15)

        fig.tight_layout()
        fig.savefig("graphs/oii_ppxf_vs_oii_lmfit.pdf")
        plt.close("all") 

    def sn_vs_v_band():
        fig, ax = plt.subplots()

        ax.scatter(data[:][:,0][:,6], data[:][:,0][:,7], color="#000000", s=10)

        for i in range(len(data[:][:,0])):
            curr_id = data[:][i,0][0]
            curr_x = data[:][i,0][6]
            curr_y = data[:][i,0][7]

            ax.annotate(int(curr_id), (curr_x, curr_y))

        ax.tick_params(labelsize=15)
        ax.set_ylabel(r'\textbf{S/N}', fontsize=15)
        ax.set_xlabel(r'\textbf{HST V-band magnitude}', fontsize=15)

        fig.tight_layout()
        fig.savefig("graphs/ppxf_sn_vs_v_band.pdf")
        plt.close("all") 

    def voigt_sigmas():
        fig, ax = plt.subplots()

        ax.scatter(data[:][:,0][:,11], data[:][:,0][:,10], color="#000000", s=10)

        for i in range(len(data[:][:,0])):
            curr_id = data[:][i,0][0]
            curr_x = data[:][i,0][11]
            curr_y = data[:][i,0][10]

            ax.annotate(int(curr_id), (curr_x, curr_y))

        ax.tick_params(labelsize=15)
        ax.set_xlabel(r'\textbf{Fitted Voigt Sigmas}', fontsize=15)
        ax.set_ylabel(r'\textbf{pPXF Voigt Sigmas}', fontsize=15)

        fig.tight_layout()
        fig.savefig("graphs/voigt_sigmas.pdf")
        plt.close("all")

    #sigma_stars_vs_sigma_oii()
    #oii_lmfit_vs_oii_ppxf()
    #sn_vs_v_band()
    #voigt_sigmas()

    #region_graphs_with_data() # calling function before this function to plot data

    # tells system to play a sound to alert that work has been finished
    os.system('afplay /System/Library/Sounds/Glass.aiff')
    personal_scripts.notifications("ppxf_fitter", "Script has finished!")

ppxf_cube_auto()
#ppxf_plots.sigma_sn()
#region_graphs_with_data()
