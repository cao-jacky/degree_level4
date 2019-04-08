import numpy as np

import catalogue_plots
import spectra_data

from uncertainties import ufloat

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

            curr_f606_w_err = ufloat(curr_f606, curr_f606_err)
            curr_f606_w_err = '{:.1uSL}'.format(curr_f606_w_err)

    # obtaining redshift and it's error from our doublet fitting
    oii_data = spectra_data.lmfit_data(cube_id)
    curr_z = oii_data['z']
    curr_z_err = oii_data['err_z_alt']

    curr_z_w_err = ufloat(curr_z, curr_z_err)
    curr_z_w_err = '{:.1uSL}'.format(curr_z_w_err)

    c = 299792.458 # speed of light in kms^-1

    # loading "a" factors in a/x model
    a_sigma_ppxf = np.load("uncert_ppxf/sigma_curve_best_values_ppxf.npy")
    a_sigma_lmfit = np.load("uncert_lmfit/sigma_curve_best_values_lmfit.npy")
    a_vel_ppxf = np.load("uncert_ppxf/vel_curve_best_values_ppxf.npy")
    a_vel_lmfit = np.load("uncert_lmfit/vel_curve_best_values_lmfit.npy")

    print(a_sigma_ppxf)

    # rounding to 4 decimal places
    #curr_z = np.around(curr_z, decimals=4)
    #curr_z_err = np.around(curr_z_err, decimals=4)

    # obtaining the velocities and velocity dispersions plus their errors
    fitted_data = np.load("data/ppxf_fitter_data.npy") 
    fd_loc = np.where(fitted_data[:,0]==cube_id)[0]
    
    if fd_loc.size == 0:
        # cubes that didn't have enough S/N for absorption line fitting with pPXF

        # calculate S/N for each cube 
        # loading x and y data 
        spec_data_x = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
                +"_cbd_x.npy")
        spec_data_y = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
                +"_cbs_y.npy")

        # S/N region is (4000,4080)*(1+z)
        # wavelength range mask
        mask_tc = (spec_data_x > 4000*(1+curr_z)) & (spec_data_x < 4080*(1+curr_z))
        mask_applied = np.where(mask_tc)[0]

        if mask_applied.size == 0:
            wr_mask = ((spec_data_x > 3727*(1+curr_z)-100) & 
                    (spec_data_x < 3727*(1+curr_z)-50))
        else:
            wr_mask = (mask_tc)
            
        stn_x = spec_data_x[wr_mask] # signal-to-noise wavelength region
        stn_y = spec_data_y[wr_mask]

        stn_mean = np.mean(stn_y)
        stn_std = np.std(stn_y)

        stn = stn_mean / stn_std # signal-to-noise
        curr_sn = stn

        sigma_oii = oii_data['sigma_gal'] # vel dispersion from lmfit
        sigma_oii = np.abs((sigma_oii/ (3727*(1+curr_z))) * c) # convert to km/s
        sigma_oii_err = (a_sigma_lmfit/curr_sn) * sigma_oii 

        vel_oii = c*np.log(1+curr_z)
        vel_oii_err = (a_vel_lmfit/curr_sn) * vel_oii

        #print(sigma_oii, sigma_oii_err, vel_oii, vel_oii_err)

        vel_oii_w_err = ufloat(vel_oii, vel_oii_err)
        vel_oii_w_err = '{:.1ufSL}'.format(vel_oii_w_err)

        sigma_oii_w_err = ufloat(sigma_oii, sigma_oii_err)
        sigma_oii_w_err = '{:.1ufSL}'.format(sigma_oii_w_err)

        print("C"+str(cube_id) + " & " + str(curr_raf_id) + " & " + str(curr_ra) + 
                " & " + str(curr_dec) + " & $" + str(curr_f606_w_err) + "$ & $" + 
                str(curr_z_w_err) + "$ & - & - & " + str(vel_oii_w_err) + " & "
                + str(sigma_oii_w_err) + " \^ \n")
    else:
        fd_curr_cube = fitted_data[fd_loc][0][0] # only need the first row of data

        curr_sn = fd_curr_cube[7] # current S/N for cube
         
        sigma_stars = fd_curr_cube[2]
        sigma_stars_err = (a_sigma_ppxf/curr_sn) * sigma_stars
        sigma_oii = fd_curr_cube[1]
        sigma_oii_err = (a_sigma_lmfit/curr_sn) * sigma_oii 

        vel_oii = c*np.log(1+curr_z)
        vel_oii_err = (a_vel_lmfit/curr_sn) * vel_oii
        vel_stars = fd_curr_cube[14]
        vel_stars_err = (a_vel_ppxf/curr_sn) * vel_stars

        # converting to shorthand uncertainties notation
        vel_stars_w_err = ufloat(vel_stars, vel_stars_err)
        vel_stars_w_err = '{:.1ufSL}'.format(vel_stars_w_err)

        vel_oii_w_err = ufloat(vel_oii, vel_oii_err)
        vel_oii_w_err = '{:.1ufSL}'.format(vel_oii_w_err)

        sigma_stars_w_err = ufloat(sigma_stars, sigma_stars_err)
        sigma_stars_w_err = '{:.1ufSL}'.format(sigma_stars_w_err)

        sigma_oii_w_err = ufloat(sigma_oii, sigma_oii_err)
        sigma_oii_w_err = '{:.1ufSL}'.format(sigma_oii_w_err)

        # print into terminal the correct line to input into LaTeX
        print("C"+str(cube_id) + " & " + str(curr_raf_id) + " & " + str(curr_ra) + 
                " & " + str(curr_dec) + " & $" + str(curr_f606_w_err) + "$ & $" + 
                str(curr_z_w_err) + "$ & $" + str(vel_stars_w_err) + "$ & $" + 
                str(sigma_stars_w_err) + "$ & $" + str(vel_oii_w_err) + "$ & $ " + 
                str(sigma_oii_w_err) + "$ \^ \n")

if __name__ == '__main__':
    data_obtainer(1804)
