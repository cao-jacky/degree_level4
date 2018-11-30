import os

import sys
sys.path.insert(0, '/Users/jackycao/Documents/Projects/scripts/')
import personal_scripts

import ppxf_fitter_kinematics_sdss
import ppxf_fitter_gas_population

import numpy as np
import matplotlib.pyplot as plt

import cube_analysis
import ppxf_plots

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

        fit_vars[i_range+1][0] = rf_vars[0]
        fit_vars[i_range+1][1] = rf_vars[1]

        fit_vars[i_range+1][2] = rf_errors[0]
        fit_vars[i_range+1][3] = rf_errors[1]

    return {'fitted_variables': fit_vars}


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
        205])

    ppxf_running = open("results/ppxf_kinematics.txt", 'w')
    ppxf_running.write("Cube ID     pPXF Reduced chi-squared    Total chi-squared       Reduced chi-squared \n")

    # we want to fit for different regions, providing an array to our region fitting
    # routine of different regions to consider for each cube
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
    data = np.zeros([len(bright_objects), 1+len(ranges), 5])

    bright_objects = np.array([0])
    catalogue = np.array([[1804]])
    
    for i_cube in range(len(bright_objects)):
        curr_obj = catalogue[i_cube]
        cube_id = int(curr_obj[0])

        data[i_cube][0][0] = cube_id

        if ((cube_id in avoid_objects) or (cube_id in cubes_to_ignore)):
            pass
        else:
            print("Currently processing cube " + str(int(cube_id)))

            # Processing the kinematic fitting
            variables = ("ppxf_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) 
                    + "_variables.npy")
            errors = ("ppxf_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) 
                    + "_errors.npy")
            if not (os.path.exists(variables) and os.path.exists(errors)):
                # fitting full standard spectrum, and only running if a numpy
                # variables file is not found - saves me the effort of waiting
                kinematic_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 
                        0, "all")

                ppxf_vars = kinematic_fit['variables']
                np.save(variables, ppxf_vars)

                ppxf_errors = kinematic_fit['errors']
                np.save(errors, ppxf_errors)
            else:
                ppxf_vars = np.load(variables)
                ppxf_errors = np.load(errors)

            # Processing the gas fitting to obtain a fitting for the OII doublet
            gas_fit = ppxf_fitter_gas_population.population_gas_sdss(cube_id, 
                    tie_balmer=False, limit_doublets=False)
            
            # using saved data, rerun analysis to find chi^2s
            ppxf_analysis = ppxf_plots.fitting_plotter(cube_id) 

            sigma_stars = ppxf_vars[1]
            data[i_cube][0][2] = sigma_stars

            sigma_stars_error = ppxf_errors[1]
            data[i_cube][0][4] = sigma_stars_error

            # now I want reaccess the lmfit file to find OII sigma
            cube_result_file = open("cube_results/cube_" + str(cube_id) + "/cube_" + 
                    str(cube_id) + "_lmfit.txt")
        
            line_count = 0 
            for crf_line in cube_result_file:
                if (line_count == 19):
                    curr_line = crf_line.split()
                    sigma_gal = float(curr_line[1])
                    sigma_gal_error = float(curr_line[3])
                if (line_count == 21):
                    curr_line = crf_line.split()
                    sigma_inst = float(curr_line[1])
                line_count += 1

            sigma = np.sqrt(sigma_gal**2 + sigma_inst**2)
            data[i_cube][0][1] = sigma
            
            data[i_cube][0][3] = sigma_gal_error
 
            # considering different ranges in the spectrum
        
            #rf = ranged_fitting(cube_id, ranges)
            rf = np.array([0])
            print(rf)
            for i_rf in range(len(rf)):
                pass

    # removing data which doesn't have an O[II] doublet for us to compare to
    sigma_doublet_vals = data[:][:,0][:,1]
    sigma_doublet_zeros = np.where(sigma_doublet_vals == 0)[0]
    data = np.delete(data, sigma_doublet_zeros, 0)
    print(data)

    # working with just one cube
    cube_id = 1129
    #ppxf_cubes(cube_id)
    #ranged_fitting(cube_id)
    #ppxf_plots.fitting_plotter(cube_id)

    # we want to plot sigma_stars vs. to sigma_OII
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

    # tells system to play a sound to alert that work has been finished
    os.system('afplay /System/Library/Sounds/Glass.aiff')
    personal_scripts.notifications("ppxf_fitter", "Script has finished!")

ppxf_cube_auto()
#ppxf_plots.sigma_sn()
