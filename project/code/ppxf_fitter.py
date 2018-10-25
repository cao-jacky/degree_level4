import os

import ppxf_fitter_kinematics_sdss
import ppxf_fitter_gas_population

import matplotlib.pyplot as plt

def ppxf_cubes(cube_id):
    file_loc = "ppxf_results"
    if not os.path.exists(file_loc):
        os.mkdir(file_loc)
    kinematics_graph = file_loc + "/cube_" + str(int(cube_id)) + "_kinematics.pdf"

    print("")
    print("Currently processing on cube " + str(int(cube_id)))
    print("Here are the results for a kinematic fitting: ")
    ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id)
    plt.ylim([-25,200])
    plt.savefig(kinematics_graph)
    print("\n")

    print("Here are the results for a mass fitting with free Balmer & [SII]: ")
    ppxf_fitter_gas_population.population_gas_sdss(cube_id, tie_balmer=False, limit_doublets=False)


ppxf_cubes(23)
