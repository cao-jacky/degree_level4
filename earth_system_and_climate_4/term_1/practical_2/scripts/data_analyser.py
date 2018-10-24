import numpy as np

import matplotlib.pyplot as plt

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

    plt.figure()
    foram_data = data_reader("d-18O.txt", 2)

    fd_x = foram_data[1:-1,0]
    fd_y = foram_data[1:-1,1]


    print(np.std(fd_y))

    plt.plot(fd_x, fd_y)
    plt.show()

plotter()

