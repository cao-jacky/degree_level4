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
    doublet_regions = np.zeros((doublet_num_lines, 4))
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

    catalogue_sorted = catalogue_sorter(cat_file_name)
    catalogue_bright = catalogue_sorted[0:10,:]
    
    for i_cube in range(len(catalogue_bright)):
        curr_row = catalogue_bright[i_cube]
    
        cube_id = int(curr_row[0])
        cube_file = "data/cubes/cube_" + str(cube_id) + ".fits" 
       
        print("Analysing cube " + str(cube_id))

        doublet_region_info = np.where( doublet_regions[:,0] == cube_id )[0]
        cube_doublet_region = doublet_regions[doublet_region_info]
        # if cannot find the region, use region for cube_23
        if (len(cube_doublet_region) == 0):
            cube_doublet_region = doublet_regions[0] 
        else:
            [cube_doublet_region] = cube_doublet_region
       
        cdr_b = int(cube_doublet_region[1])
        cdr_e = int(cube_doublet_region[2])
        doublet_range = [cdr_b, cdr_e]

        sky_file = "data/skyvariance_csub.fits"
        peak_loc = int(cube_doublet_region[3])
         
        #if (cube_id == 23):
            #cube_reader.analysis(cube_file, sky_file, doublet_range, peak_loc)
        
        cube_reader.analysis(cube_file, sky_file, doublet_range, peak_loc)
        

multi_cube_reader("data/catalog.fits")
