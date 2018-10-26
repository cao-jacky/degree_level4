import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import cube_reader
import multi_cubes

def highest_sn():
    cube_data_file = open("data/cube_doublet_regions.txt")
    cd_num_lines = sum(1 for line in open("data/cube_doublet_regions.txt")) - 1
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

            cube_file = ("results/cube_" + str(cube_id) + "/cube_" + str(cube_id) + 
                "_fitting.txt")
            cube_file_data = open(cube_file)

            cb_file_lines = sum(1 for line in open("results/cube_" + str(cube_id) + 
                "/cube_" + str(cube_id) + "_fitting.txt")) - 1

            cb_file_count = 0
            for cb_line in cube_file_data:
                if (cb_file_count == (cb_file_lines-1)):
                    cb_curr_line = cb_line.split()

                    sn_value = cb_curr_line[-1]
                    usable_cubes[usable_count][1] = sn_value

                cb_file_count += 1

            usable_count += 1

    usable_cubes = usable_cubes[usable_cubes[:,1].argsort()[::-1]]
    #print(usable_cubes)
    return(usable_cubes)

def sky_noise_cut():
    sky_file_loc = "data/skyvariance_csub.fits"
    sky_file_data = cube_reader.sky_noise(sky_file_loc)

    # finding indices where peaks are still viewable past the specifie cut heigh
    cut_height = 10
    sky_data_cut = np.nonzero(sky_file_data > 10)[0] 
    return (sky_data_cut)

def find_nearest(array, value):
    """ Find nearest value is an array """
    idx = (np.abs(array-value)).argmin()
    return idx

def data_cube_analyser(cube_id):
    cube_x_data = np.load("results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbd_x.npy")
    cube_y_data = np.load("results/cube_" + str(int(cube_id)) + "/cube_" + 
        str(int(cube_id)) + "_cbs_y.npy")

    sky_noise = sky_noise_cut(cube_id)

    # let's cut out the region of the O[II] doublet to around where I think the 
    # absorption lines end
    abs_region = [5300, 6300]
    abs_region_indexes = [find_nearest(cube_x_data, x) for x in abs_region]
    ari = abs_region_indexes
    
    abs_region_x = cube_x_data[ari[0]:ari[1]]
    abs_region_y = cube_y_data[ari[0]:ari[1]]

    # we have a region now we want to plot log(flux) vs mag

    cat_file = "data/catalog.fits"
    cat_data = multi_cubes.catalogue_sorter(cat_file)
    cat_curr_cube = np.where(cat_data[:,0] == cube_id)[0]

    # cat: col 51 contains f606, col 37 uses the magnitude for stars
    curr_cube_cat_data = cat_data[cat_curr_cube]
    cat_mag = curr_cube_cat_data[50]
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    plt.figure()
    plt.plot(cube_x_data, sky_noise, linewidth=0.5, color="#000000")
    plt.axhline(10, linewidth=0.5, color="#00c853")
    plt.savefig("graphs/sanity_checks/sky_spectra.pdf")

    plt.figure()
    plt.plot(cube_x_data, cube_y_data, linewidth=0.5, color="#000000")
    plt.savefig("graphs/sanity_checks/cube_" + str(int(cube_id)) + "_spectra.pdf")
    
    plt.figure()
    plt.plot(abs_region_x, abs_region_y, linewidth=0.5, color="#000000")
    plt.savefig("graphs/sanity_checks/cube_" + str(int(cube_id)) + "_abs_spectra.pdf")
   
def cube_noise(cube_id):
    cube_file_name = "data/cubes/cube_" + str(int(cube_id)) + ".fits" 
    cube_file = cube_reader.read_file(cube_file_name)
    
    image_data = cube_file[1]
    collapsed_data = cube_reader.spectrum_creator(cube_file_name)
    
    id_shape = np.shape(image_data)
    cd_shape = collapsed_data['shape']
     
    last_pixel = image_data[:,-1,-1]
    
    # create noise region based off how big the collapsed_data shape
    if ( cd_shape[1] <= (id_shape[1]/2) ):
        nr_shape = [10,10]
    else: 
        nr_shape = [5,5]

    nr_loc = np.array(id_shape[1:]) - np.array(nr_shape)
    noise_data = image_data[:,nr_loc[1]:id_shape[1], nr_loc[0]:id_shape[0]]

    nr_noise = np.zeros(np.shape(noise_data)[0])
    for i_ax in range(np.shape(noise_data)[0]):
        col_data = noise_data[i_ax][:]
        nr_noise[i_ax] = np.sum(col_data)

    noise = np.std(nr_noise) / np.sqrt(id_shape[1]**2/nr_shape[1]**2)
    return {'noise_data': nr_noise, 'noise_value': noise}

def vband_graphs():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    cube_data_file = open("data/cube_doublet_regions.txt")
    cd_num_lines = sum(1 for line in open("data/cube_doublet_regions.txt")) - 1
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
    usable_cubes = np.zeros((len(usable_count),5))

    # usable_cubes structure
    # [0] : cube id
    # [1] : v-band mag from catalogue for object
    # [2] : median for absorption range
    # [3] : mean for absorption range
    # [4] : signal to noise

    usable_count = 0
    for i_cube in range(len(cube_data)):
        usability = int(cube_data[i_cube][-1])

        if ( usability == 1 ):
            cube_id = int(cube_data[i_cube][0])
            usable_cubes[usable_count][0] = cube_id

            cube_x_data = np.load("results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_cbd_x.npy")
            cube_y_data = np.load("results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_cbs_y.npy")

            abs_region = [5500, 7000]
            abs_region_indexes = [find_nearest(cube_x_data, x) for x in abs_region]
            ari = abs_region_indexes

            abs_region_x = cube_x_data[ari[0]:ari[1]]
            abs_region_y = cube_y_data[ari[0]:ari[1]]

            abs_flux_median = np.abs(np.median(abs_region_y))
            abs_flux_average = np.abs(np.average(abs_region_y))

            usable_cubes[usable_count][2] = -2.5 * np.log10(abs_flux_median)
            usable_cubes[usable_count][3] = -2.5 * np.log10(abs_flux_average)

            cat_file = "data/catalog.fits"
            cat_data = multi_cubes.catalogue_sorter(cat_file)
            cat_curr_cube = np.where(cat_data[:,0] == cube_id)[0]

            # column 51 in the catalogue contains f606nm data which is about V-band
            # v-band has a midpint wavelength of ~551nm
            vband_mag = cat_data[cat_curr_cube][0][50]
            usable_cubes[usable_count][1] = vband_mag

            # we want to select a region to calculat the signal to noise
            cube_file = ("results/cube_" + str(cube_id) + "/cube_" + str(cube_id) + 
                "_lmfit.txt")
            cube_file_data = open(cube_file)

            cb_file_lines = sum(1 for line in open("results/cube_" + str(cube_id) + 
                "/cube_" + str(cube_id) + "_lmfit.txt")) - 1

            cb_file_count = 0
            for cb_line in cube_file_data:
                if (cb_file_count == 20 ):
                    cb_curr_line = cb_line.split()
                    z = float(cb_curr_line[1])

                cb_file_count += 1
            
            # plotting s/n vs mag
            abs_region2 = [3700, 4500, 4250]
            abs_region2z = [x*(1+z) for x in abs_region2]
            abs_region2_indexes = [find_nearest(cube_x_data, x) for x in abs_region2z]
            ar2i = abs_region2_indexes

            # cube y data for the absorption region - this is our signal
            ar_y = cube_y_data[ar2i[0]:ar2i[1]]
            ar_x = cube_x_data[ar2i[0]:ar2i[1]] 

            # signal and noise
            ar_signal = np.median(ar_y)
            ar_noise = cube_noise(cube_id)['noise_value']

            signal_noise = np.abs(ar_signal / ar_noise)
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
                plt.savefig("graphs/sanity_checks/cubes/absorption_region_" + 
                        str(int(cube_id)) + ".pdf")
            
            abs_region_graphs()

            usable_cubes[usable_count][4] = signal_noise
             
            usable_count += 1

    #print(usable_cubes)

    plt.figure()
    plt.scatter(usable_cubes[:,2], usable_cubes[:,1], s=10, color="#000000")
    plt.title(r'\textbf{V-band mag vs. flux-mag}', fontsize=13)        
    plt.xlabel(r'\textbf{flux-mag}', fontsize=13)
    plt.ylabel(r'\textbf{V-band mag}', fontsize=13)
    plt.savefig("graphs/sanity_checks/vband_vs_flux.pdf")

    fig, ax = plt.subplots()
    ax.scatter(usable_cubes[:,1], usable_cubes[:,4], s=10, color="#000000")

    # plotting a generic line of best fit
    #ax.plot(np.unique(usable_cubes[:,1]), np.poly1d(np.polyfit(usable_cubes[:,1],
        #usable_cubes[:,4], 1))(np.unique(usable_cubes[:,1])))

    cube_ids = usable_cubes[:,0]
    for i, txt in enumerate(cube_ids):
        ax.annotate(int(txt), (usable_cubes[i][1], usable_cubes[i][4]))

    ax.set_title(r'\textbf{S/N vs. V-band mag }', fontsize=13)        
    ax.set_xlabel(r'\textbf{V-band mag}', fontsize=13)
    ax.set_ylabel(r'\textbf{S/N}', fontsize=13)
    ax.set_ylim([0,30])
    plt.savefig("graphs/sanity_checks/sn_vs_vband.pdf")

#print(highest_sn())
#data_cube_analyser(468)

vband_graphs()
#cube_noise(23)
