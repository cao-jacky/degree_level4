import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import multi_cubes

def data_matcher(cat_file_name, doublet_file):
    """ matching our doublet text file which contains the details of run through
    cubes """

    catalogue_sorted = multi_cubes.catalogue_sorter(cat_file_name)

    doublet_regions_file = open(doublet_file)
    doublet_num_lines = sum(1 for line in open(str(doublet_file))) - 1
    doublet_regions = np.zeros((doublet_num_lines, 5))
    file_row_count = 0
    for file_line in doublet_regions_file:
        file_line = file_line.split()
        if (file_row_count == 0):
            pass
        else:
            for file_col in range(len(file_line)):
                doublet_regions[file_row_count-1][file_col] = file_line[file_col]

        file_row_count += 1 

    doublet_regions_file.close()

    # running through data and finding required information
    usable_locs = np.where(doublet_regions[:,-1] == 1)[0]
    
    # producing three columns: cube_id, catalogue redshift, model redshift, model
    # redshift error
    usable_data = np.zeros((len(usable_locs), 4))
    for i_cube in range(len(usable_locs)):
        cube_index = usable_locs[i_cube]

        cube_id = int(doublet_regions[cube_index][0])
        usable_data[i_cube][0] = cube_id

        # looking at data from catalogue: col 44 has redshift data
        cat_cube_loc = np.where(catalogue_sorted[:,0] == cube_id)[0]
        cat_cube_data = catalogue_sorted[cat_cube_loc][0]

        cat_cube_rdst = cat_cube_data[43]
        usable_data[i_cube][1] = cat_cube_rdst

        # looking at our generated model results
        final_results_loc = ("results/cube_" + str(cube_id) + "/cube_" + str(cube_id) 
            + "_lmfit.txt")
        final_results_file = open(final_results_loc) 
        
        # 15th line in the final results file contains the optimal model redshift
        curr_line = 0
        for fr_line in final_results_file:
            if (curr_line == 20):
                rdst_line = fr_line.split()
                rdst_val = rdst_line[1]
                rdst_err = rdst_line[3]

                usable_data[i_cube][2] = rdst_val
                usable_data[i_cube][3] = rdst_err
            curr_line += 1

        final_results_file.close() 

    return usable_data

def chisq(model, data, data_err):
    csq = (data-model)**2 / data_err**2
    csq_final = np.sum(csq)

    redcsq = csq_final / (len(csq))
    return {'chisq': csq_final, 'redchisq': redcsq}

def plots(file_catalogue, file_doublets):
    data = data_matcher(file_catalogue, file_doublets)
    
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
        fig.savefig("graphs/redshift.pdf")

    redshift()

file_catalogue  = "data/catalog.fits"
file_doublets   = "data/cube_doublet_regions.txt"
plots(file_catalogue, file_doublets)



