from astropy.io import fits

import numpy as np

import os
import re

import cube_reader

def catalogue_file_reader(cat_file_name):
    """ reads the cube data catalogue """
    fits_file       = fits.open(cat_file_name)

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

def cube_analyser(cube_id): 
    cube_file = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/" + "cube_"
        + str(cube_id) + ".fits")
    sky_file = "data/skyvariance_csub.fits" 
    cube_reader.analysis(cube_file, sky_file)

def multi_cube_reader(catalogue_array):
    """ takes sorted catalogue (sorted by 775nm filter) and then runs through the
    specified integer amount of data """

    catalogue = np.load(catalogue_array)

    # repeating extractor conditions: sort by probability then consider only top 300
    # objects
    catalogue = catalogue[catalogue[:,8].argsort()]
    catalogue = catalogue[0:300,:]
     
    # cubes.txt file structure
    # [0]: cube id according to sextractor
    # [1]: doublet region begin in Å
    # [2]: doublet region end in Å
    # [3]: the location of the doublet peak (approx)
    # [4]: 'usability' of the data: 0 is for no, 1 is for yes (usable), 2 is for unsure
    cubes_file = open("data/cubes.txt")
    cubes_file_num_lines = sum(1 for line in open("data/cubes.txt")) - 1
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

    # selecting objects which are brighter than 23.0 magnitude
    bright_objects = np.where(catalogue[:,5] < 32.0)[0]
    avoid_objects = np.array([357,695,1393,1504,773,1212,1522,1052,609,1656,58,1373,
        893,742,1293,1572,865,7,1681,761,1475,4,699,1444,1600,819,905,1206,585,468,
        1529,1092,299,904,356,620,65,1511,1633,1417,114,282,1722,532,62,1682,833,
        1223,295,268,1367,1626,1728,661,1726,1484,1745,785,1218,1624,558,418,193,
        1590,697,1160,667,897,861,153,1430,873,1312,463,271,677,1055,128,342,1707,
        1203,374,196,1535,1440,1335,774,146,565,1282,184,975,1226,1478,253,1356,
        502,684,1150,574,572,448,1331,125,1054,457,1589,605,17,1222,837,892,618,281,
        707])

    np.save("data/avoid_objects", avoid_objects)

    for i_obj in bright_objects: 
        curr_obj = catalogue[i_obj]
        obj_id = int(curr_obj[0]) 

        if (obj_id in avoid_objects):
            print("Avoiding cube " + str(obj_id))
            pass
        else:
            data_dir = 'cube_results/cube_' + str(obj_id)
            if not os.path.exists(data_dir):
                print("Working with cube " + str(obj_id))
                cube_analyser(obj_id)
            else:
                print("Skipping cube " + str(obj_id))
                pass
if __name__ == '__main__':
    multi_cube_reader("data/matched_catalogue.npy")
