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
    kin_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id) 

    """
    print("Processing mass fitting with free Balmer & [SII]: ")
    ppxf_fitter_gas_population.population_gas_sdss(cube_id, tie_balmer=False, limit_doublets=False)
    print("")
    print("Processing mass fitting with tied Balmer & [SII]: ")
    ppxf_fitter_gas_population.population_gas_sdss(cube_id, tie_balmer=True, limit_doublets=True)"""
    print("\n")

    return {'kinematic_fitting': kin_fit}

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
    ppxf_running.write("Cube ID     Reduced chi-squared\n")

    """
    for i_cube in range(len(bright_objects)):
        curr_obj = catalogue[i_cube]
        cube_id = int(curr_obj[0])

        if ((cube_id in avoid_objects) or (cube_id in cubes_to_ignore)):
            pass
        else:
            fit = ppxf_cubes(cube_id)
            ppxf_plots.chi_squared_cal(cube_id)

            kin_fit_chi2 = fit['kinematic_fitting']['reduced_chi2']
            ppxf_running.write(str(cube_id) + "     " + str(kin_fit_chi2) + "\n")
            """

    cube_id = 5
    ppxf_cubes(cube_id)
    ppxf_plots.chi_squared_cal(cube_id)


ppxf_cube_auto()
