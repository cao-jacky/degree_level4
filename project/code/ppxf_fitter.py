import os

import ppxf_fitter_kinematics_sdss
import ppxf_fitter_gas_population

import numpy as np
import matplotlib.pyplot as plt

import cube_analysis

def ppxf_cubes(cube_id):
    print("")
    print("Currently processing on cube " + str(int(cube_id)))
    print("Processing kinematic fitting: ")
    ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id) 

    print("Processing mass fitting with free Balmer & [SII]: ")
    ppxf_fitter_gas_population.population_gas_sdss(cube_id, tie_balmer=False, limit_doublets=False)
    print("")
    print("Processing mass fitting with tied Balmer & [SII]: ")
    ppxf_fitter_gas_population.population_gas_sdss(cube_id, tie_balmer=True, limit_doublets=True)
    print("\n")

def ppxf_cube_auto():

    cubes_to_process = cube_analysis.highest_sn()
    for i_cube in range(len(cubes_to_process)):
        cube_id = int(cubes_to_process[i_cube][0])

        if (cube_id in np.array([1162, 112, 722, 764, 769, 760, 1469])):
            pass
        else:
            ppxf_cubes(cube_id)

ppxf_cube_auto()
