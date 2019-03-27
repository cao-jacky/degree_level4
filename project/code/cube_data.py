import numpy as np

import catalogue_plots
import spectra_data

def data_obtainer(cube_id):
    # uses the cube ID to return:
    # Cube ID, RAF ID, RA, Dec, HST F606, z, V_*, sig_*, V_OII, sig_OII

    # load the combined catalogue
    file_read = catalogue_plots.read_cat("data/matched_catalogues.fits")
    catalogue_data = file_read['data']

    # locating row where catalogue data is stored
    #cat_loc = np.where(catalogue_data[:,375]==cube_id)
    #print(cat_loc)

    for i_object in range(len(catalogue_data)):
        curr_object = catalogue_data[i_object]
        curr_id = curr_object[375] 

        if curr_id == cube_id:
            curr_raf_id = curr_object[7]
            curr_ra = curr_object[1]
            curr_dec = curr_object[2]
            curr_f606 = curr_object[64]
            curr_f606_err = curr_object[65]

    # obtaining redshift and it's error from our doublet fitting
    oii_data = spectra_data.lmfit_data(cube_id)
    curr_z = oii_data['z']
    curr_z_err = oii_data['err_z_alt']
    
    # rounding to 4 decimal places
    curr_z = np.around(curr_z, decimals=4)
    curr_z_err = np.around(curr_z_err, decimals=4)

    # obtaining the velocities and velocity dispersions plus their errors
    fitted_data = np.load("data/ppxf_fitter_data.npy") 
    fd_loc = np.where(fitted_data[:,0]==cube_id)[0]
    
    if fd_loc.size == 0:
        print("C"+str(cube_id) + " & " + str(curr_raf_id) + " & " + str(curr_ra) + 
                " & " + str(curr_dec) + " & $" + str(curr_f606) + "\pm" + 
                str(curr_f606_err) + "$ & $" + str(curr_z) + "\pm" + str(curr_z_err) +
                "$ & - & - & - & - \^ \n")
    else:
        fd_curr_cube = fitted_data[fd_loc][0][0] # only need the first row of data

        # loading "a" factors in a/x model
        a_sigma_ppxf = np.load("uncert_ppxf/sigma_curve_best_values_ppxf.npy")
        a_sigma_lmfit = np.load("uncert_lmfit/sigma_curve_best_values_lmfit.npy")
        a_vel_ppxf = np.load("uncert_ppxf/vel_curve_best_values_ppxf.npy")
        a_vel_lmfit = np.load("uncert_lmfit/vel_curve_best_values_lmfit.npy")

        curr_sn = fd_curr_cube[7] # current S/N for cube
         
        sigma_stars = fd_curr_cube[2]
        sigma_stars_err = (a_sigma_ppxf/curr_sn) * sigma_stars
        sigma_oii = fd_curr_cube[1]
        sigma_oii_err = (a_sigma_lmfit/curr_sn) * sigma_oii 

        # rounding values
        sigma_stars = int(np.around(sigma_stars, decimals=0))
        sigma_stars_err = int(np.around(sigma_stars_err, decimals=0)[0])
        sigma_oii = int(np.around(sigma_oii, decimals=0))
        sigma_oii_err = int(np.around(sigma_oii_err, decimals=0)[0])

        c = 299792.458 # speed of light in kms^-1
        vel_oii = c*np.log(1+curr_z)
        vel_oii_err = (a_vel_lmfit/curr_sn) * vel_oii
        vel_stars = fd_curr_cube[14]
        vel_stars_err = (a_vel_ppxf/curr_sn) * vel_stars

        # rounding values
        vel_stars = int(np.around(vel_stars, decimals=0))
        vel_stars_err = int(np.around(vel_stars_err, decimals=0)[0])
        vel_oii = int(np.around(vel_oii, decimals=0))
        vel_oii_err = int(np.around(vel_oii_err, decimals=0)[0])

        # print into terminal the correct line to input into LaTeX
        print("C"+str(cube_id) + " & " + str(curr_raf_id) + " & " + str(curr_ra) + 
                " & " + str(curr_dec) + " & $" + str(curr_f606) + "\pm" + 
                str(curr_f606_err) + "$ & $" + str(curr_z) + "\pm" + str(curr_z_err) +
                "$ & $" + str(vel_stars) + "\pm" + str(vel_stars_err) + "$ & $" + 
                str(sigma_stars) + "\pm" + str(sigma_stars_err) + "$ & $" + 
                str(vel_oii) + "\pm" + str(vel_oii_err) + "$ & $ " + str(sigma_oii) + 
                "\pm" + str(sigma_oii_err) + "$ \^ \n")

if __name__ == '__main__':
    data_obtainer(1804)
