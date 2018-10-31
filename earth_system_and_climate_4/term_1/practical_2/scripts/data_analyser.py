import numpy as np

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
    fig, ax1 = plt.subplots()
    fd_x = foram_data[1:-1,0]
    fd_y = foram_data[1:-1,1]
    ax1.set_xlabel(r'\textbf{Age (Myr)}', fontsize=13)
    ax1.set_ylabel(r'\textbf{d18O}', fontsize=13)
    ax1.set_xlim([0,5])
    ax1.scatter(fd_x, fd_y, s=2.5, color="#000000")

    z = np.polyfit(fd_x, fd_y, 1)
    p = np.poly1d(z)
    #ax1.plot(fd_x,p(fd_x),"r--")

    orbit_data = data_reader("orbit_data.txt", 5)
    od_x = orbit_data[1:-1,0]

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # eccentricity
    od_e = orbit_data[1:-1,1]
    ax2.scatter(od_x, od_e, s=2.5, color="#f44336")
    ax2.set_ylabel(r'\textbf{Eccentricity}', fontsize=13)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
 
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

    #plt.show()

plotter()

