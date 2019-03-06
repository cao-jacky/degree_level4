import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

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

    ax.tick_params(labelsize=20)
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
    uc = np.array([1804])
    print(uc)
    for i_cube in range(len(uc)):
        cube_id = int(uc[i_cube])

        print("cube_"+str(cube_id)+": ")
        
        seg_overlay(cube_id) # creating image of galaxy with segmentation map overlayed
        spectra(cube_id) # creating a spectra

        # Remaking old plots
        #cube_reader.analysis("/Volumes/Jacky_Cao/University/level4/project/"+
                #"cubes_better/cube_"+str(cube_id)+".fits", 
                #"data/skyvariance_csub.fits")
    

if __name__ == '__main__':
    file_catalogue  = "data/matched_catalogue.npy"
    file_cubes      = "data/cubes.txt"
    #plots(file_catalogue, file_cubes)

    auto_runner()
