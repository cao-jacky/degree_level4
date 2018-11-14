import numpy as np
from numpy import genfromtxt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

def data_reader(file_name, array_cols):
    open_file_name = open(file_name)
    open_file_name_2 = open(file_name)
    file_num_lines = sum(1 for line in open_file_name_2) - 1

    data_store = np.zeros((file_num_lines+1, array_cols))

    file_curr_line = 0
    for i_row in open_file_name:
        if (file_curr_line == 0):
            pass
        else: 
            curr_line_split = i_row.split()
            for i_col in range(len(curr_line_split)):
                data_store[file_curr_line][i_col] = curr_line_split[i_col]
            
        file_curr_line += 1

    return data_store

def plotter():
    foram_data = data_reader("d-18O.txt", 2)

    # foram
    plt.figure(1)
    plt.subplot(211)
    fd_x = foram_data[1:-1,0]
    fd_y = foram_data[1:-1,1]
    plt.ylabel(r'\textbf{$\delta^{18}$O}', fontsize=15)
    plt.xlim([0,1])
    plt.scatter(fd_x, fd_y, s=2.5, color="#000000")
    plt.plot(fd_x, fd_y, linewidth=0.5, color="#000000", alpha=0.2)

    plt.subplot(212)
    plt.xlabel(r'\textbf{Age (Myr)}', fontsize=15)
    plt.ylabel(r'\textbf{$\delta^{18}$O}', fontsize=15)
    plt.xlim([4,5])
    plt.scatter(fd_x, fd_y, s=2.5, color="#000000")
    plt.plot(fd_x, fd_y, linewidth=0.5, color="#000000", alpha=0.2)

    z = np.polyfit(fd_x, fd_y, 1)
    p = np.poly1d(z)
    #ax1.plot(fd_x,p(fd_x),"r--")

    orbit_data = data_reader("orbit_data.txt", 5)
    od_x = orbit_data[1:-1,0]

    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # eccentricity
    #od_e = orbit_data[1:-1,1]
    #ax2.scatter(od_x, od_e, s=2.5, color="#f44336")
    #ax2.set_ylabel(r'\textbf{Eccentricity}', fontsize=13)

    plt.tick_params(labelsize=15)
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.show()
    plt.savefig("foram_data.pdf", dpi=500)
 
    # obliquity
    plt.figure()
    od_o = orbit_data[1:-1,2]
    plt.xlabel(r'\textbf{Age (Myr)}', fontsize=13)
    plt.ylabel(r'\textbf{Obliquity (radians)}', fontsize=13)
    plt.scatter(od_x, od_o, s=5)

    # perihelion
    plt.figure()
    od_p = orbit_data[1:-1,3]
    plt.xlabel(r'\textbf{Age (Myr)}', fontsize=13)
    plt.ylabel(r'\textbf{Perihelion longitude (radians)}', fontsize=13)
    plt.scatter(od_x, od_p, s=5)

    # precessional index
    plt.figure()
    od_pi = orbit_data[1:-1,4]
    plt.xlabel(r'\textbf{Age (Myr)}', fontsize=13)
    plt.ylabel(r'\textbf{Precessional index}', fontsize=13)
    plt.scatter(od_x, od_pi, s=5)

    plt.close("all")

    #plt.show()

def d18O_redfit():
    data = genfromtxt('d-18O_redfit.csv', delimiter=',')

    frequency = data[:,0]
    power = data[:,1]
    chisq95 = data[:,5]
    
    mask = np.where(power < chisq95)[0]
    masked_frequency_regions = np.split(frequency, mask)
    masked_power_regions = np.split(power, mask)

    plt.figure()
    plt.plot(frequency, power, linewidth=0.5, color="#000000")

    for i in range(len(masked_frequency_regions)):
        x_masked = masked_frequency_regions[i]
        y_masked = masked_power_regions[i]
        plt.plot(x_masked, y_masked, linewidth=0.5, color="#388e3c")

    plt.plot(frequency, chisq95, linewidth=0.5, color="#e53935")

    plt.xlabel(r'\textbf{Frequency (Myr$^{-1}$)}', fontsize=15)
    plt.ylabel(r'\textbf{Power}', fontsize=15)
    plt.tick_params(labelsize=15)

    plt.xlim([0,60])
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("d18O_redfit.pdf", dpi=500)
    plt.close("all")

    frequency_peaks = np.array([10.3882, 24.3173, 42.164])
    wavelengths = 1/frequency_peaks

    print(wavelengths)

def heatmaps():
    data = genfromtxt('wavelet_analysis_eccentricity.csv', delimiter=',')

    log2_scale = data[1:,0]
    x_scale = data[0:1,2:-1]

    heatmap_data = data[1:,2:-1]
    print(np.shape(heatmap_data))
    print(heatmap_data)

    plt.figure()
    #plt.axis([np.min(x_scale), np.max(x_scale), np.min(log2_scale), 
        #np.max(log2_scale)])
    #plt.yticks(log2_scale)

    plt.imshow(heatmap_data)

    plt.xlabel(r'\textbf{Frequency (Myr$^{-1}$)}', fontsize=15)
    plt.ylabel(r'\textbf{Power}', fontsize=15)

    plt.tick_params(labelsize=15)
    #plt.yticks(log2_scale)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    plt.savefig("wa_eccentricity.pdf", dpi=500)
    plt.close("all")



#plotter()
#d18O_redfit()
heatmaps()

