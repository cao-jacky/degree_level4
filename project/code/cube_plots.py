import numpy as np

import matplotlib.pyplot as plt

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

def data_matcher(cat_file_name, doublet_file):
    """ matching our doublet text file which contains the details of run through
    cubes """

    catalogue_sorted = catalogue_sorter(cat_file_name)
