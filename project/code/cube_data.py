import numpy as np

import catalogue_plots

def data_obtainer(cube_id):
    # uses the cube ID to return:
    # Cube ID, RAF ID, RA, Dec, HST F606, z, V_*, sig_*, V_OII, sig_OII

    # load the combined catalogue
    file_read = catalogue_plots.read_cat("data/matched_catalogues.fits")
    catalogue_data = file_read['data']

    print(catalogue_data)

    # locating row where catalogue data is stored
    #cat_loc = np.where(catalogue_data[:,375]==cube_id)
    #print(cat_loc)

    for i_object in range(len(catalogue_data)):
        curr_object = catalogue_data[i_object]

        curr_id = curr_object[375] 

        if curr_id == cube_id:
            print(curr_id)

if __name__ == '__main__':
    data_obtainer(1804)
