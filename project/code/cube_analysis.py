import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches
from matplotlib.ticker import StrMethodFormatter

from collections import OrderedDict

import cube_reader
import spectra_data
import ppxf_fitter
import cube_data

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

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

    # let's cut out the region of the [OII] doublet to around where I think the 
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

    plt.figure()
    plt.plot(cube_x_data, cube_y_data, linewidth=0.5, color="#000000")
    plt.savefig("graphs/sanity_checks/cube_" + str(int(cube_id)) + "_spectra.pdf")
    
    plt.figure()
    plt.plot(abs_region_x, abs_region_y, linewidth=0.5, color="#000000")
    plt.savefig("graphs/sanity_checks/cube_" + str(int(cube_id)) + "_abs_spectra.pdf") 

def f_doublet(c, i1, i2, sigma_gal, z, sigma_inst):
    """ function for Gaussian doublet """  
    dblt_mu = [3727.092, 3729.875] # the actual non-redshifted wavelengths
    l1 = dblt_mu[0] * (1+z)
    l2 = dblt_mu[1] * (1+z)

    x = 3727 * (1+z)

    sigma = np.sqrt(sigma_gal**2 + sigma_inst**2)

    norm = (sigma*np.sqrt(2*np.pi))
    term1 = ( i1 / norm ) * np.exp(-(x-l1)**2/(2*sigma**2))
    term2 = ( i2 / norm ) * np.exp(-(x-l2)**2/(2*sigma**2)) 
    return (c*x + term1 + term2)

def luminosity_flux(z, oii_flux):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    distance = cosmo.luminosity_distance(z).value * (3.09*(10**22)) 
    luminosity = (oii_flux * 4 * np.pi * (distance**2))
    return luminosity

def graphs(): 
    catalogue = np.load("data/matched_catalogue.npy")
    catalogue = catalogue[catalogue[:,8].argsort()]
    catalogue = catalogue[0:300,:] 

    bright_objects = np.where(catalogue[:,5] < 32.0)[0]

    avoid_objects = np.load("data/avoid_objects.npy")
    more_useless = np.array([474,167,1101,1103,744])
    #more_useless = np.array([0])

    # array to store cube id and signal to noise value]
    usable_cubes = np.zeros((len(bright_objects)-len(avoid_objects)-len(more_useless)
        +1,14))
    
    # usable_cubes structure
    # [0] : cube id
    # [1] : v-band mag from catalogue for object
    # [2] : median for absorption range
    # [3] : mean for absorption range
    # [4] : signal to noise
    # [5] : total counts for image
    # [6] : noise for image 
    # [7] : B-I colour
    # [8] : I1 flux from model fitting
    # [9] : I2 flux from model fitting
    # [10] : c from model fitting
    # [11] : sigma_gal from model fitting
    # [12] : sigma_inst from model fitting
    # [13] : z from model fitting

    usable_count = 0
    for i_cube in bright_objects:
        curr_obj = catalogue[i_cube]
        cube_id = int(curr_obj[0]) 

        if (cube_id in avoid_objects or cube_id in more_useless or 
                usable_count==np.shape(usable_cubes)[0]):
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
            # parameters from lmfit
            lm_params = spectra_data.lmfit_data(cube_id)
            c = lm_params['c']
            i1 = lm_params['i1']
            i2 = lm_params['i2']
            sigma_gal = lm_params['sigma_gal']
            z = lm_params['z']
            sigma_inst = lm_params['sigma_inst']

            usable_cubes[usable_count][10] = c
            usable_cubes[usable_count][8] = i1 
            usable_cubes[usable_count][9] = i2
            usable_cubes[usable_count][11] = sigma_gal
            usable_cubes[usable_count][13] = z
            usable_cubes[usable_count][12] = sigma_inst
            
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
            usable_cubes[usable_count][4] = signal_noise
            #print(cube_id, ar_signal, ar_noise, signal_noise)
            
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
                ax1.set_title(r'\textbf{galaxy: median}', fontsize=20)    
                ax1.set_xlabel(r'\textbf{Pixels}', fontsize=20)
                ax1.set_ylabel(r'\textbf{Pixels}', fontsize=20) 

                ax2.imshow(im_coll_data['sum'], cmap='gray_r')
                ax2.set_title(r'\textbf{galaxy: sum}', fontsize=20)        
                ax2.set_xlabel(r'\textbf{Pixels}', fontsize=20)
                ax2.set_ylabel(r'\textbf{Pixels}', fontsize=20)

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

            bband_mag = float(cat_data[cat_curr_cube][0][14])
            iband_mag = float(cat_data[cat_curr_cube][0][15])
            usable_cubes[usable_count][7] = bband_mag - iband_mag

            usable_count += 1

    #print(usable_cubes)

    unusable_cubes = ppxf_fitter.ignore_cubes()

    # --------------------------------------------------#

    # V-BAND VS. FLUX MAG
    fig, ax = plt.subplots()
    ax.scatter(usable_cubes[:,2], usable_cubes[:,1], s=7, color="#000000")

    cube_ids = usable_cubes[:,0]
    for i, txt in enumerate(cube_ids):
        ax.annotate(int(txt), (usable_cubes[i][2], usable_cubes[i][1]))

    #ax.set_title(r'\textbf{V-band mag vs. flux-mag}', fontsize=13)        
    ax.set_xlabel(r'\textbf{flux-mag}', fontsize=13)
    ax.set_ylabel(r'\textbf{V-band mag}', fontsize=13)
    plt.tight_layout()
    plt.savefig("graphs/sanity_checks/vband_vs_flux.pdf")

    # --------------------------------------------------#

    # S/N VS. V-BAND MAG
    fig, ax = plt.subplots()
    #ax.scatter(usable_cubes[:,1], usable_cubes[:,4], s=20, color="#000000")

    # plotting the usable cubes
    for i in range(len(usable_cubes[:,0])):
        curr_cube = int(usable_cubes[:,0][i]) 
        if curr_cube in unusable_cubes['ac']:
            ax.scatter(usable_cubes[:,1][i], usable_cubes[:,4][i], s=20, 
                    color="#ffa000", alpha=1.0, marker="x", 
                    label=r'\textbf{Not usable}')
        if curr_cube in unusable_cubes['ga']:
            ax.scatter(usable_cubes[:,1][i], usable_cubes[:,4][i], s=20, 
                    color="#ffa000", alpha=1.0, marker="x") 
        if curr_cube not in unusable_cubes['ac'] and usable_cubes[:,1][i] < 25.0:
            cube_data.data_obtainer(curr_cube) # creating LaTeX prepared table entry
            
            ax.scatter(usable_cubes[:,1][i], usable_cubes[:,4][i], s=20, 
                    color="#00c853", alpha=1.0, marker="o", zorder=3, 
                    label=r'\textbf{Usable}')

    ax.fill_between(np.linspace(25,28,100), 0, 100, alpha=0.2, zorder=0, 
            facecolor="#ffcdd2")

    #ax.set_title(r'\textbf{S/N vs. V-band mag }', fontsize=13)       
    ax.tick_params(labelsize=20)
    ax.set_xlabel(r'\textbf{HST V-band magnitude}', fontsize=20)
    ax.set_ylabel(r'\textbf{Spectrum S/N}', fontsize=20)
    ax.invert_xaxis()
    ax.set_yscale('log')
    ax.set_ylim([0.9, 100])
    ax.set_xlim([26,20])
   
    # manually setting x-tick labels to be 1 dpm
    
    vband_x = np.array([26.0, 24.0, 22.0, 20.0])
    ax.set_xticks(vband_x) # locations of ticks
    ax.set_xticklabels([r'\textbf{'+str(vband_x[0])+'}',
            r'\textbf{'+str(vband_x[1])+'}',r'\textbf{'+str(vband_x[2])+'}',
            r'\textbf{'+str(vband_x[3])+'}'])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys(), loc='lower right', prop={'size': 15})

    plt.tight_layout()
    plt.savefig("graphs/sn_vs_vband.pdf",bbox_inches="tight")

    # --------------------------------------------------#

    usable_cubes_no_oii = usable_cubes
    cubes_to_ignore = ppxf_fitter.ignore_cubes()['ac']

    cubes_to_ignore_indices = []

    for i_cube in range(len(cubes_to_ignore)):
        curr_cube = cubes_to_ignore[i_cube]
        #loc = np.where(usable_cubes[:,0] == curr_cube)[0].item()
        loc = np.where(usable_cubes[:,0] == curr_cube)[0]
        cubes_to_ignore_indices.append(loc)

    cubes_to_ignore_indices = np.sort(np.asarray(cubes_to_ignore_indices))[::-1]
    
    for i_cube in range(len(cubes_to_ignore_indices)):
        index_to_delete = cubes_to_ignore_indices[i_cube]
        usable_cubes_no_oii = np.delete(usable_cubes_no_oii, index_to_delete, axis=0)

    oii_flux = f_doublet(usable_cubes_no_oii[:,10], usable_cubes_no_oii[:,8], 
            usable_cubes_no_oii[:,9], usable_cubes_no_oii[:,11], 
            usable_cubes_no_oii[:,13], usable_cubes_no_oii[:,12])
    
    # we want to convert the flux into proper units
    oii_flux = oii_flux * (10**(-20)) # 10**-20 Angstrom-1 cm-2 erg s-1
    oii_flux = oii_flux / (10**(-10) * 10**(-4))
    
    # [OII] FLUX VS. GALAXY COLOUR
    fig, ax = plt.subplots()
    ax.scatter(usable_cubes_no_oii[:,7], oii_flux, s=7, color="#000000")

    cube_ids = usable_cubes_no_oii[:,0]
    for i, txt in enumerate(cube_ids):
        ax.annotate(int(txt), (usable_cubes_no_oii[i][7], oii_flux[i]), alpha=0.2)

    #ax.set_title(r'\textbf{S/N vs. V-band mag }', fontsize=13)       
    ax.tick_params(labelsize=20)
    ax.set_xlabel(r'\textbf{Galaxy Colour (B-I)}', fontsize=20)
    ax.set_ylabel(r'\textbf{[OII] Flux}', fontsize=20)
    plt.tight_layout()
    plt.savefig("graphs/oii_flux_vs_colour.pdf",bbox_inches="tight")
    plt.close("all")

    # [OII] VELOCITY DISPERSION VS. STELLAR MAG


    # REDSHIFT DISTRIBUTION OF [OII] EMITTERS 
    fig, ax = plt.subplots()
    
    ax.hist(usable_cubes_no_oii[:,13], facecolor="#000000")
     
    ax.tick_params(labelsize=20)
    ax.set_xlabel(r'\textbf{Redshift}', fontsize=20)
    ax.set_ylabel(r'\textbf{Number of galaxies}', fontsize=20)
    plt.tight_layout()
    plt.savefig("graphs/redshift_distribution_oii_emitters.pdf")
    plt.close("all")

    # OII LUMINOSITY VS. REDSHIFT
    # let's pick out the flux which is the smallest
    oii_flux_smallest = np.min(oii_flux)
    ofs_redshifts = np.arange(0,1.6,0.01)
    ofs_luminosity = luminosity_flux(ofs_redshifts, oii_flux_smallest)

    cubes_luminosity = luminosity_flux(usable_cubes_no_oii[:,13], oii_flux)

    print(len(usable_cubes_no_oii[:,13]))

    fig, ax = plt.subplots()
    ax.plot(ofs_redshifts, ofs_luminosity, linewidth=1.5, color="#9e9e9e")
    ax.scatter(usable_cubes_no_oii[:,13], cubes_luminosity, s=20, color="#000000")

    cube_ids = usable_cubes_no_oii[:,0]
    for i, txt in enumerate(cube_ids):
        pass
        #ax.annotate(int(txt), (usable_cubes_no_oii[i][13], cubes_luminosity[i]), 
            #alpha=0.2)

    ax.fill_between(np.linspace(0.0,0.3,100), 0.007*10**44,1.7*10**55, alpha=0.2, 
            zorder=0, facecolor="#ffcdd2")
  
    ax.tick_params(labelsize=20)
    ax.set_xlabel(r'\textbf{Redshift}', fontsize=20)
    ax.set_ylabel(r'\textbf{[OII] Luminosity (W)}', fontsize=20)
    ax.set_yscale('log')
    ax.set_xlim([0.0, 1.5])
    ax.set_ylim((0.5*10**45,0.3*10**52))
    plt.tight_layout()
    plt.savefig("graphs/o_ii_luminosity_vs_redshift.pdf",bbox_inches="tight")
    plt.close("all")
 
if __name__ == '__main__':
    #print(highest_sn())
    #data_cube_analyser(1804)

    graphs()

    #cube_noise()
    pass
