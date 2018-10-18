from astropy.io import fits

import numpy as np

import re

import cube_reader

def catalogue_file_reader(cat_file_name):
    """ reads the cube data catalogue """
    fits_file       = fits.open(cat_file_name)

    file_header     = fits_file[0].header

    table_header    = fits_file[1].header
    table_data      = fits_file[1].data
    return {'table_header': table_header, 'table_data': table_data} 

def catalogue_sorter(cat_file_name):
    """ processing and then sorting data by the filter with 775nm """
    file_details = catalogue_file_reader(cat_file_name)

    # manually read the catalogue file to find where column data for 775nm is
    cat_header = file_details['table_header']
    cat_data = file_details['table_data']

    # f775 data is contained in column 52, we want to convert AstroPy data to Numpy
    # once we convert data into numpy array, this column will become column 51 because 
    # of pythonic indexing schemes
    cat_np_data = np.zeros((len(cat_data), len(cat_data[0])))
    for i_row in range(len(cat_data)):
        curr_row = cat_data[i_row]
        for i_col in range(len(curr_row)):
            data_to_store = curr_row[i_col] 

            if (str(data_to_store).isalpha() == True):
                if (str(data_to_store) == "nan"):
                    data_to_store = 0
                else:
                    data_to_store = ord(str(data_to_store))

            cat_np_data[i_row][i_col] = data_to_store

    cat_np_data = cat_np_data[cat_np_data[:,51].argsort()]

    return cat_np_data

def multi_cube_reader(cat_file_name):
    """ takes sorted catalogue (sorted by 775nm filter) and then runs through the
    specified integer amount of data """
   
    doublet_regions_file = open("data/cube_doublet_regions.txt")
    doublet_num_lines = sum(1 for line in open("data/cube_doublet_regions.txt")) - 1
    doublet_regions = np.zeros((doublet_num_lines, 3))
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

    print(doublet_regions)


multi_cube_reader("data/catalog.fits")
