import numpy as np

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
    print(usable_cubes)

highest_sn()
