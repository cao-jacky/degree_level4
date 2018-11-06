import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches

import cube_reader

from astropy.io import fits

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

def highest_sn():
    cube_data_file = open("data/cubes.txt")
    cd_num_lines = sum(1 for line in open("data/cubes.txt")) - 1
    cube_data = np.zeros((cd_num_lines, 5))

    file_row_count = 0
    for file_line in cube_data_file:
        file_line = file_line.split()
        if (file_row_count == 0):
            pass
        else:
            for file_col in range(len(file_line)):
                cube_data[file_row_count-1][file_col] = file_line[file_col]
        file_row_count += 1 

    cube_data_file.close()

    # array to store cube id and signal to noise value
    usable_count = np.where(cube_data[:,-1] == 1)[0]
    usable_cubes = np.zeros((len(usable_count),2))

    usable_count = 0
    for i_cube in range(len(cube_data)):
        usability = int(cube_data[i_cube][-1])

        if ( usability == 1 ):
            cube_id = int(cube_data[i_cube][0])
            usable_cubes[usable_count][0] = cube_id

            cube_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + str(cube_id)
                    + "_fitting.txt")
            cube_file_data = open(cube_file)

            cb_file_lines = sum(1 for line in open("cube_results/cube_" + str(cube_id)
                + "/cube_" + str(cube_id) + "_fitting.txt")) - 1

            cb_file_count = 0
            for cb_line in cube_file_data:
                if (cb_file_count == (cb_file_lines-1)):
                    cb_curr_line = cb_line.split()

                    sn_value = cb_curr_line[-1]
                    usable_cubes[usable_count][1] = sn_value

                cb_file_count += 1

            usable_count += 1

    usable_cubes = usable_cubes[usable_cubes[:,1].argsort()[::-1]]
    return(usable_cubes)

def sky_noise_cut():
    sky_file_loc = "data/skyvariance_csub.fits"
    sky_file_data = cube_reader.sky_noise(sky_file_loc)

    # finding indices where peaks are still viewable past the specifie cut heigh
    cut_height = 10
    sky_data_cut = np.nonzero(sky_file_data > 10)[0] 
    return (sky_data_cut)

def cube_noise():
    cube_noise_file = "data/cube_noise.fits"
    cube_noise_file = fits.open(cube_noise_file)

    cube_noise_data = cube_noise_file[1].data 
    noise = np.sum(np.abs(cube_noise_data))
    return {'noise_value': noise, 'spectrum_noise': cube_noise_data}

def find_nearest(array, value):
    """ Find nearest value is an array """
    idx = (np.abs(array-value)).argmin()
    return idx

def data_cube_analyser(cube_id):
    cube_x_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbd_x.npy")
    cube_y_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbs_y.npy")

    sky_noise = sky_noise_cut()

    # let's cut out the region of the O[II] doublet to around where I think the 
    # absorption lines end
    abs_region = [5300, 6300]
    abs_region_indexes = [find_nearest(cube_x_data, x) for x in abs_region]
    ari = abs_region_indexes
    
    abs_region_x = cube_x_data[ari[0]:ari[1]]
    abs_region_y = cube_y_data[ari[0]:ari[1]]

    # we have a region now we want to plot log(flux) vs mag

    cat_file = "data/matched_catalogue.npy"
    cat_data = np.load(cat_file)
    cat_curr_cube = np.where(cat_data[:,0] == cube_id)[0]

    # cat: col 5 contains f606
    curr_cube_cat_data = cat_data[cat_curr_cube][0]
    cat_mag = curr_cube_cat_data[5]
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    plt.figure()
    plt.plot(cube_x_data, cube_y_data, linewidth=0.5, color="#000000")
    plt.savefig("graphs/sanity_checks/cube_" + str(int(cube_id)) + "_spectra.pdf")
    
    plt.figure()
    plt.plot(abs_region_x, abs_region_y, linewidth=0.5, color="#000000")
    plt.savefig("graphs/sanity_checks/cube_" + str(int(cube_id)) + "_abs_spectra.pdf") 

def vband_graphs():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    catalogue = np.load("data/matched_catalogue.npy")
    catalogue = catalogue[catalogue[:,8].argsort()]
    catalogue = catalogue[0:300,:] 

    bright_objects = np.where(catalogue[:,5] < 26.0)[0]

    avoid_objects = np.load("data/avoid_objects.npy")
    #more_useless = np.array([474,167,1101,1103,744])
    more_useless = np.array([0])

    # array to store cube id and signal to noise value]
    usable_cubes = np.zeros((len(bright_objects)-len(avoid_objects)-len(more_useless)+1
        ,7))

    # usable_cubes structure
    # [0] : cube id
    # [1] : v-band mag from catalogue for object
    # [2] : median for absorption range
    # [3] : mean for absorption range
    # [4] : signal to noise
    # [5] : total counts for image
    # [6] : noise for image 

    usable_count = 0
    for i_cube in bright_objects:
        curr_obj = catalogue[i_cube]
        cube_id = int(curr_obj[0]) 

        if (cube_id in avoid_objects or cube_id in more_useless):
            pass
        else:
            usable_cubes[usable_count][0] = cube_id

            cube_x_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" 
                    + str(int(cube_id)) + "_cbd_x.npy")
            cube_y_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" 
                    + str(int(cube_id)) + "_cbs_y.npy")

            abs_region = [5500, 7000]
            abs_region_indexes = [find_nearest(cube_x_data, x) for x in abs_region]
            ari = abs_region_indexes

            abs_region_x = cube_x_data[ari[0]:ari[1]]
            abs_region_y = cube_y_data[ari[0]:ari[1]]

            abs_flux_median = np.abs(np.nanmedian(abs_region_y))
            abs_flux_average = np.abs(np.nanmean(abs_region_y))

            usable_cubes[usable_count][2] = -2.5 * np.log10(abs_flux_median)
            usable_cubes[usable_count][3] = -2.5 * np.log10(abs_flux_average)

            cat_file = "data/matched_catalogue.npy"
            cat_data = np.load(cat_file)
            cat_curr_cube = np.where(cat_data[:,0] == cube_id)[0]

            # column 5 in the catalogue contains f606nm data which is about V-band
            # v-band has a midpint wavelength of ~551nm
            vband_mag = cat_data[cat_curr_cube][0][5]
            usable_cubes[usable_count][1] = vband_mag

            # we want to select a region to calculate the signal to noise
            cube_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) 
                    + "_lmfit.txt")
            cube_file_data = open(cube_file)

            cb_file_lines = sum(1 for line in open("cube_results/cube_" + str(cube_id) 
                + "/cube_" + str(cube_id) + "_lmfit.txt")) - 1

            cb_file_count = 0
            for cb_line in cube_file_data:
                if (cb_file_count == 20 ):
                    cb_curr_line = cb_line.split()
                    z = float(cb_curr_line[1])

                cb_file_count += 1
            
            # plotting s/n vs mag
            lower_lambda = (1+z)*3700
            upper_lambda = (1+z)*4500

            #-absorption region mask
            arm = (lower_lambda < cube_x_data) & (cube_x_data < upper_lambda) 
            
            # cube y data for the absorption region - this is our signal
            ar_y = cube_y_data[arm]
            ar_x = cube_x_data[arm] 

            cube_noise_data = cube_noise()
            spectrum_noise = cube_noise_data['spectrum_noise']

            # signal and noise
            ar_signal = np.median(ar_y)
            ar_noise = np.sum(spectrum_noise)

            signal_noise = np.abs(ar_signal/ar_noise)
            print(cube_id, ar_signal, ar_noise, signal_noise)
            
            def abs_region_graphs():
                plt.figure()
                plt.plot(cube_x_data, cube_y_data, linewidth=0.5, color="#000000")
                plt.plot(ar_x, ar_y, linewidth=0.3, color="#d32f2f")

                plt.axhline(ar_signal, linewidth=0.5, color="#212121")
                plt.axhline(ar_signal+ar_noise, linewidth=0.5, color="#212121", 
                        alpha=0.75)
                plt.axhline(ar_signal-ar_noise, linewidth=0.5, color="#212121", 
                        alpha=0.75)

                plt.ylim([-1000,5000])

                #plt.title(r'\textbf{'+str(ar_signal)+' '+str(ar_noise)+' '+
                        #str(signal_noise)+'}', fontsize=13)
                plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
                plt.ylabel(r'\textbf{Flux}', fontsize=13)
                plt.savefig("graphs/sanity_checks/absregions/absorption_region_" + 
                        str(int(cube_id)) + ".pdf")
        
            #abs_region_graphs()

            def graphs_collapsed():
                cube_file = ("/Volumes/Jacky_Cao/University/level4/project/" + 
                        "cubes_better/cube_" + str(cube_id) + ".fits")
                im_coll_data = cube_reader.image_collapser(cube_file)
         
                f, (ax1, ax2)  = plt.subplots(1, 2)
            
                ax1.imshow(im_coll_data['median'], cmap='gray_r') 
                ax1.set_title(r'\textbf{galaxy: median}', fontsize=13)    
                ax1.set_xlabel(r'\textbf{Pixels}', fontsize=13)
                ax1.set_ylabel(r'\textbf{Pixels}', fontsize=13) 

                ax2.imshow(im_coll_data['sum'], cmap='gray_r')
                ax2.set_title(r'\textbf{galaxy: sum}', fontsize=13)        
                ax2.set_xlabel(r'\textbf{Pixels}', fontsize=13)
                ax2.set_ylabel(r'\textbf{Pixels}', fontsize=13)

                gal = patches.Rectangle((gal_region[0],gal_region[1]),
                        gal_region[3]-gal_region[0],gal_region[2]-gal_region[1],
                        linewidth=1, edgecolor='#b71c1c', facecolor='none')
                noise = patches.Rectangle((noise_region[0],noise_region[1]),
                        noise_region[3]-noise_region[0],noise_region[2]-noise_region[1]
                        , linewidth=1, edgecolor='#1976d2', facecolor='none')

                # Add the patch to the Axes
                ax1.add_patch(gal)
                ax1.add_patch(noise)

                f.subplots_adjust(wspace=0.4)
                f.savefig("graphs/sanity_checks/stacked/stacked" + 
                        str(int(cube_id)) + ".pdf")

            #graphs_collapsed()

            usable_cubes[usable_count][4] = signal_noise
            usable_count += 1

    fig, ax = plt.subplots()
    ax.scatter(usable_cubes[:,2], usable_cubes[:,1], s=10, color="#000000")

    cube_ids = usable_cubes[:,0]
    for i, txt in enumerate(cube_ids):
        ax.annotate(int(txt), (usable_cubes[i][2], usable_cubes[i][1]))

    ax.set_title(r'\textbf{V-band mag vs. flux-mag}', fontsize=13)        
    ax.set_xlabel(r'\textbf{flux-mag}', fontsize=13)
    ax.set_ylabel(r'\textbf{V-band mag}', fontsize=13)
    plt.savefig("graphs/sanity_checks/vband_vs_flux.pdf")

    fig, ax = plt.subplots()
    ax.scatter(usable_cubes[:,1], usable_cubes[:,4], s=10, color="#000000")

    cube_ids = usable_cubes[:,0]
    for i, txt in enumerate(cube_ids):
        ax.annotate(int(txt), (usable_cubes[i][1], usable_cubes[i][4]))

    ax.set_title(r'\textbf{S/N vs. V-band mag }', fontsize=13)        
    ax.set_xlabel(r'\textbf{V-band mag}', fontsize=13)
    ax.set_ylabel(r'\textbf{S/N}', fontsize=13)
    ax.invert_xaxis()
    ax.set_yscale('log')
    plt.savefig("graphs/sn_vs_vband.pdf")

    fig, ax = plt.subplots()
    ax.scatter(usable_cubes[:,1], usable_cubes[:,6], s=10, color="#000000")

    cube_ids = usable_cubes[:,0]
    for i, txt in enumerate(cube_ids):
        ax.annotate(int(txt), (usable_cubes[i][1], usable_cubes[i][6]))

    ax.set_title(r'\textbf{Image S/N vs. V-band mag }', fontsize=13)        
    ax.set_xlabel(r'\textbf{V-band mag}', fontsize=13)
    ax.set_ylabel(r'\textbf{S/N for image}', fontsize=13)
    plt.savefig("graphs/sanity_checks/image_sn_vs_vband.pdf")

    plt.close("all")

#print(highest_sn())
#data_cube_analyser(1804)

vband_graphs()

#cube_noise()
