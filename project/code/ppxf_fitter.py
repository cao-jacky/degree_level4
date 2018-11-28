import os

import ppxf_fitter_kinematics_sdss
import ppxf_fitter_gas_population

import numpy as np
import matplotlib.pyplot as plt

import cube_analysis
import ppxf_plots

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

def ranged_fitting(cube_id):
    """ fitting pPXF for different ranges """
    # fitting for the full, original spectrum
    original = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all")

    ori_vars = original['variables']
    ori_errors = original['errors']

    # we want to fit for three different regions
    ranges = np.array([[3540, 3860],[3860, 4180],[4180, 4500]])

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

    print(fit_vars)

def ppxf_cube_auto():

    catalogue = np.load("data/matched_catalogue.npy")
    catalogue = catalogue[catalogue[:,8].argsort()]
    catalogue = catalogue[0:300,:] 

    bright_objects = np.where(catalogue[:,5] < 24.0)[0]

    avoid_objects = np.load("data/avoid_objects.npy")
    cubes_to_ignore = np.array([97,139,140,152,157,159,178,1734,1701,1700,1689,
        1592,1580,1564,1562,1551,1514,1506,1454,1446,1439,1418,1388,1355,1353,1267,
        1227,1210,1198,1187,1159,1157,1137,1132,1121,217,222,317,338,339,343,361,395,
        407,421,438,458,459,481,546,551,582,592,651,659,665,687,710,720,737,744,754,
        789,790,814,834,859,864,878,879,914,965,966,982,1008,1017,1033,1041,1045,
        1063,1068,1114,1162, 112, 722, 764, 769, 760, 1469, 733, 1453, 723, 378,
        135, 474, 1103])

    ppxf_running = open("results/ppxf_kinematics.txt", 'w')
    ppxf_running.write("Cube ID     pPXF Reduced chi-squared    Total chi-squared       Reduced chi-squared \n")

    """
    for i_cube in range(len(bright_objects)):
        curr_obj = catalogue[i_cube]
        cube_id = int(curr_obj[0])

        if ((cube_id in avoid_objects) or (cube_id in cubes_to_ignore)):
            pass
        else:
            ppxf_fit = ppxf_cubes(cube_id)
            chi_squared = ppxf_plots.fitting_plotter(cube_id)

            kin_fit_chi2 = ppxf_fit['kinematic_fitting']['reduced_chi2']

            tot_chi2 = chi_squared['chi2']
            red_chi2 = chi_squared['redchi2']

            ppxf_running.write(str(cube_id) + "     " + str(kin_fit_chi2) + "     " + 
                    str(tot_chi2) + "     " + str(red_chi2) + "\n")
            
    """
    cube_id = 1804
    #ppxf_cubes(cube_id)
    ranged_fitting(cube_id)
    #ppxf_plots.fitting_plotter(cube_id)

ppxf_cube_auto()
#ppxf_plots.sigma_sn()
