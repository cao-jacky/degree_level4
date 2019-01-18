import ppxf_fitter_kinematics_sdss

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

import os

def ppxf_uncertainty(cubes, runs):
    cubes = cubes # array containing cubes to parse over
    to_run = runs # number of times to run the random generator

    # 1st dimension: for every cube there is an array
    # 2nd dimension: there are same number of rows as to_run variable
    # 3rd dimension: columns to store data
    #   [0] : new signal value [= signal + perturbation]
    #   [1] : new sigma produced
    #   [2] : (sigma_best - sigma_new) / sigma_best
    #   [3] : new signal to noise value 
    #   [4] : new velocity produced
    #   [5] : (vel_best - vel_new) / vel_best
    #   [6] : sigma error
    #   [7] : velocity error
    #   [8] : (sigma_best - sigma_new) 

    data = np.zeros([len(cubes),to_run,9])
    
    for i_cube in range(len(cubes)):
        cube_id = cubes[i_cube]
        
        # running the best fit fitting routine 
        best_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all")

        z = best_fit['redshift']

        best_noise = best_fit['noise_original']

        best_x = best_fit['x_data']
        best_y = best_fit['y_data']

        best_variables = best_fit['variables']
        best_sigma = best_variables[1]
        best_vel = best_variables[0]

        # want to consider between CaH and Hdelta, the range to consider (rtc) is
        #rtc = np.array([3969.588, 4101.89]) * (1+z) 
        rtc = np.array([4000, 4080]) 
        rtc_mask = ((best_x > rtc[0]) & (best_x < rtc[1]))

        best_y_masked = best_y[rtc_mask]
        best_noise_masked = best_noise[rtc_mask]

        noise_median = np.median(best_noise_masked)

        best_sn = best_y_masked / best_noise_masked
        average_best_sn = np.average(best_sn)
        
        # the median of the data and the average of the noise should be similar
        n1 = np.std(best_y_masked)
        n2 = np.average(best_noise_masked)
        print(n1, n2, average_best_sn, np.average(best_y_masked/np.var(best_y_masked)))
        original_y = best_fit['y_data_original']
        galaxy_spectrum = original_y 

        n_std = np.average(best_noise_masked)
 
        for i in range(to_run):
            print("working with " + str(cube_id) + " and index " + str(i))

            # generating a random noise distribution using a mean of 0 and the 
            # standard deviation of the original galaxy spectrum within a region
            random_noise = np.random.normal(0, n_std, len(galaxy_spectrum))
            if (i > int(3/4 * to_run)):
                random_noise = random_noise
            else:
                random_noise = 10 * random_noise

            galaxy_spectrum = galaxy_spectrum + random_noise
            print(n_std, np.std(random_noise))

            new_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 
                    galaxy_spectrum, "all")

            new_variables = new_fit['variables']
            new_sigma = new_variables[1]
            new_vel = new_variables[0]

            sigma_ratio = np.abs(best_sigma-new_sigma) / best_sigma
            sigma_diff = np.abs(best_sigma-new_sigma)

            vel_ratio = np.abs(best_vel-new_vel) / best_vel
            
            new_x = new_fit['x_data']
            new_y = new_fit['y_data']
             
            new_mask = ((new_x > rtc[0]) & (new_x < rtc[1]))

            new_x = new_x[new_mask]
            new_y = new_y[new_mask]

            #non_scaled_y = new_fit['non_scaled_y'][new_mask]
            
            new_signal = new_y            
            new_noise = np.std(new_y)

            new_sn_total = new_signal / new_noise
            new_sn = np.average(new_sn_total)
 
            print(new_sigma, best_sigma, new_sn)

            errors = new_fit['errors']
            error_sigma = errors[1]
            error_vel = errors[0]

            data[i_cube][i][0] = np.median(new_signal) # new signal
            data[i_cube][i][1] = new_sigma # new sigma
            data[i_cube][i][2] = sigma_ratio # sigma ratio

            data[i_cube][i][3] = new_sn # signal to noise

            data[i_cube][i][4] = new_vel # new velocity
            data[i_cube][i][5] = vel_ratio # velocity ratio

            data[i_cube][i][6] = error_sigma # sigma error
            data[i_cube][i][7] = error_vel # velocity error

            data[i_cube][i][8] = sigma_diff # sigma difference

            plt.figure() 
            plt.plot(best_x[new_mask], best_y[new_mask], linewidth=0.5, 
                    color="#8bc34a")
            plt.plot(new_x, new_y, linewidth=0.5, color="#000000")

            plt.xlabel(r'\textbf{S/N}', fontsize=15)
            plt.ylabel(r'\textbf{$\frac{\Delta \sigma}{\sigma_{best}}$}', fontsize=15)

            plt.tight_layout()

            uncert_ppxf_dir = "uncert_ppxf/cube_"+str(cube_id)
            if not os.path.exists(uncert_ppxf_dir):
                os.mkdir(uncert_ppxf_dir)
            plt.savefig(uncert_ppxf_dir + "/cube_"+str(cube_id)+ 
                    "_" + str(i) + ".pdf")
            plt.close("all") 
        
        print(data[i_cube][:][:])
        np.save("uncert_ppxf/cube_"+str(cube_id)+"/cube_"+str(cube_id)+"_ppxf_perts",
                data[:,i_cube][:][:])

    np.save("data/sigma_vs_sn_data", data)

    def sigma_vs_sn():
        plt.figure()
        #plt.scatter(best_sn, best_sigma/best_sigma, color="#b71c1c", s=10)
        for i in range(len(data[:])):
            plt.scatter(data[i][:,3], data[i][:,2], c=np.random.rand(3,), s=10)
        #plt.ylim([np.min(data[:,:,3]), np.max(data[:,:,3])])

        plt.xlabel(r'\textbf{S/N}', fontsize=15)
        plt.ylabel(r'\textbf{$\frac{\Delta \sigma}{\sigma_{best}}$}', fontsize=15)

        plt.tight_layout()
        plt.savefig("uncert_ppxf/sigma_vs_sn.pdf")
        plt.close("all") 

    def sigma_vel_vs_sn():
        plt.figure()
        for i in range(len(data[:])):
            plt.scatter(data[i][:,3], data[i][:,5], c=np.random.rand(3,), s=10)
     
        plt.xlabel(r'\textbf{S/N}', fontsize=15)
        plt.ylabel(r'\textbf{$\frac{\Delta \sigma_{vel}}{\sigma_{vel_{best}}}$}', 
                fontsize=15)

        plt.tight_layout()
        plt.savefig("uncert_ppxf/sigma_vel_vs_sn.pdf")
        plt.close("all")

    def sn_vs_sigma():
        plt.figure()
        #plt.scatter(best_sn, best_sigma/best_sigma, color="#b71c1c", s=10)
        for i in range(len(data[:])):
            plt.scatter(data[i][:,2], data[i][:,3], c=np.random.rand(3,), s=10)
        #plt.ylim([np.min(data[:,:,3]), np.max(data[:,:,3])])

        plt.ylabel(r'\textbf{S/N}', fontsize=15)
        plt.xlabel(r'\textbf{$\frac{\Delta \sigma}{\sigma_{best}}$}', fontsize=15)

        plt.tight_layout()
        plt.savefig("uncert_ppxf/sn_vs_sigma.pdf")
        plt.close("all") 

    def sn_vs_sigma_diff():
        plt.figure()
        #plt.scatter(best_sn, best_sigma/best_sigma, color="#b71c1c", s=10)
        for i in range(len(data[:])):
            plt.scatter(data[i][:,8], data[i][:,3], c=np.random.rand(3,), s=10)
        #plt.ylim([np.min(data[:,:,3]), np.max(data[:,:,3])])

        plt.ylabel(r'\textbf{S/N}', fontsize=15)
        plt.xlabel(r'\textbf{$\Delta \sigma$}', fontsize=15)

        plt.tight_layout()
        plt.savefig("uncert_ppxf/delta_sn_vs_sigma.pdf")
        plt.close("all") 

    sigma_vs_sn()
    sigma_vel_vs_sn()
    sn_vs_sigma()
    sn_vs_sigma_diff()

cubes = np.array([1804, 765, 5, 1, 767, 1578, 414, 1129, 286, 540])
#cubes = np.array([1804])
runs = 300
ppxf_uncertainty(cubes, runs)
