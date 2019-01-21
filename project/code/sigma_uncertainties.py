import ppxf_fitter_kinematics_sdss
import spectra_data

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

from lmfit import Parameters, Model

import os

import sys
sys.path.insert(0, '/Users/jackycao/Documents/Projects/scripts/')
import personal_scripts

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
            plt.ylabel(r'\textbf{Flux}', fontsize=15)

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

    np.save("data/ppxf_uncert_data", data)

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

def ppxf_graphs():
    data = np.load("data/ppxf_uncert_data.npy")

    # colours list
    colours = spectra_data.colour_list()

    def sn_vs_delta_sigma_sigma():
        total_bins1 = 400
        X = data[:,:,3] # x-axis should be fractional error
        
        bins = np.linspace(X.min(), X.max(), total_bins1)
        delta = bins[1]-bins[0]
        idx  = np.digitize(X,bins)

        #####
        # S/N vs. delta(sigma)/sigma
        plt.figure()
        for i in range(len(data[:])): 
            plt.scatter(data[i][:,2], data[i][:,3], c=colours[i], s=10, alpha=0.2)
            
        # running median calculator
        Y_sigma = data[:,:,2] # y-axis should be signal to noise
        running_median1 = [np.median(Y_sigma[idx==k]) for k in range(total_bins1)]

        rm1 = np.array(running_median1)        
        y_data1 = (bins-delta/2)

        plt.plot(rm1, y_data1, c="#000000", lw=1.5, alpha=0.7)

        idx = np.isfinite(rm1) # mask to mask out finite values
        fitted_poly = np.poly1d(np.polyfit(rm1[idx], y_data1[idx], 4))
        t = np.linspace(np.min(y_data1), np.max(y_data1), 200)
        plt.plot(fitted_poly(t), t, c="#d32f2f", lw=1.5, alpha=0.8)

        plt.ylabel(r'\textbf{S/N}', fontsize=15)
        plt.xlabel(r'\textbf{$\Delta \sigma / \sigma_{best}$}', fontsize=15)

        #plt.xlim([10**(-4),100])
        #plt.yscale('log')
        plt.tight_layout()
        plt.savefig("uncert_ppxf/sn_vs_delta_sigma_sigma.pdf")
        plt.close("all") 

        #####
        # S/N vs. delta(sigma_vel)/sigma_vel
        def sn_del_sigma_vel():
            plt.figure()
            for i in range(len(data[:])):
                plt.scatter(data[i][:,5], data[i][:,3], c=colours[i], s=10, alpha=0.2) 

            Y_sigma_vel = data[:,:,5]
            running_median2 = [np.median(Y_sigma_vel[idx==k]) for k in 
                    range(total_bins1)]
            plt.plot(running_median2, bins-delta/2, c="#000000", lw=1.5, alpha=0.7)

            plt.ylabel(r'\textbf{S/N}', fontsize=15)
            plt.xlabel(r'\textbf{$\Delta \sigma_{vel} / \sigma_{vel_{best}}$}', 
                    fontsize=15)

            plt.xlim([-np.min(data[:,:,5]),0.0021])
            #plt.xscale('log')
            plt.tight_layout()
            plt.savefig("uncert_ppxf/sn_vs_d_sigma_vel_sigma_vel.pdf")
            plt.close("all")

    def sn_vs_delta_sigma():
        # S/N vs. delta(sigma)
        total_bins2 = 400
        X_sigma = data[:,:,3]
        
        bins = np.linspace(X_sigma.min(), X_sigma.max(), total_bins2)
        delta = bins[1]-bins[0]
        idx  = np.digitize(X_sigma,bins)

        plt.figure()
        for i in range(len(data[:])):
            plt.scatter(data[i][:,8], data[i][:,3], c=colours[i], s=10, alpha=0.2)

        Y_sn = data[:,:,8]
        running_median3 = [np.median(Y_sn[idx==k]) for k in range(total_bins2)]
        plt.plot(running_median3, bins-delta/2, c="#000000", lw=1.5, alpha=0.7)

        plt.tick_params(labelsize=15)
        plt.ylabel(r'\textbf{S/N}', fontsize=15)
        plt.xlabel(r'\textbf{$\Delta \sigma$}', fontsize=15)

        #plt.ylim([10**(-8),100])
        #plt.yscale('log')
        plt.tight_layout()
        plt.savefig("uncert_ppxf/sn_vs_delta_sigma.pdf")
        plt.close("all")

    def sn_vs_frac_error():
        # S/N vs. fractional error
        total_bins2 = 400
        X_sigma = data[:,:,3]
        
        bins = np.linspace(X_sigma.min(), X_sigma.max(), total_bins2)
        delta = bins[1]-bins[0]
        idx  = np.digitize(X_sigma,bins)

        plt.figure()
        for i in range(len(data[:])):
            plt.scatter(data[i][:,3], data[i][:,2], c=colours[i], s=10, alpha=0.2)

        # Running median for data
        Y_sn = data[:,:,2]
        running_median3 = [np.median(Y_sn[idx==k]) for k in range(total_bins2)]

        rm3 = np.array(running_median3)
        xd = bins-delta/2

        #plt.plot(xd, rm3, c="#000000", lw=1.5, alpha=0.7)
        plt.scatter(xd, rm3, c="#000000", s=10, alpha=0.7)

        # Fitting an a/x line to the data
        def curve(x, a):
            return (a/x)

        curve_params = Parameters()
        curve_params.add('a', value=1)
        curve_model = Model(curve)

        idx = np.isfinite(rm3) # mask to mask out finite values
        curve_result = curve_model.fit(rm3[idx], x=xd[idx], params=curve_params)
    
        curve_bf = curve_result.best_fit
        plt.plot(xd[idx], curve_bf, c="#d32f2f", lw=1.5, alpha=0.8)
 
        plt.tick_params(labelsize=15)
        plt.xlabel(r'\textbf{S/N}', fontsize=15)
        plt.ylabel(r'\textbf{${|\Delta \sigma|}/{\sigma_{best}}$}', fontsize=15)

        plt.tight_layout()
        plt.savefig("uncert_ppxf/frac_error_vs_sn.pdf")
        plt.close("all")

    sn_vs_delta_sigma_sigma()
    sn_vs_delta_sigma()
    sn_vs_frac_error()

    os.system('afplay /System/Library/Sounds/Glass.aiff')
    personal_scripts.notifications("ppxf_plots","Reprocessed plots have been plotted!")

def lmfit_uncertainty(cubes, runs):
    data = np.zeros([len(cubes), runs, 4]) 

    # 1st dimension: one array per cube
    # 2nd dimension: same number of rows as runs variable
    # 3rd dimension: columns to store data
    #   [0] : new signal value [= signal + perturbation]
    #   [1] : new sigma produced
    #   [2] : (sigma_best - sigma_new) / sigma_best
    #   [3] : new signal to noise value 

    # Looping over all of the provided cubes
    for i_cube in range(len(cubes)):
        cube_id = cubes[i_cube]

        # Pulling the parameters for the best fit
        bd = spectra_data.lmfit_data(cube_id) # best data
        best_sigma = bd['sigma_gal']
        best_z = bd['z']

        # Load (non-redshifted) wavelength and flux data
        x_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
                +"_corr_x.npy")
        y_data = np.load("cube_results/cube_"+str(cube_id)+"/cube_"+str(cube_id)
                +"_cps_y.npy")
 
        # Define doublet region to consider to be 3500Å to 3700Å
        dr_mask = ((x_data > 3500) & (x_data < 3700))
        
        x_masked = x_data[dr_mask]
        y_masked = y_data[dr_mask]

        # Standard deviation of the data in region before the doublet
        data_std = np.std(y_masked)

        # Fake data created based on the best fitting parameters
        x_fake = np.linspace(3500,3800,600) * (1+best_z)
        y_fake = spectra_data.f_doublet(x_fake, bd['c'], bd['i1'], bd['i2'], 
                bd['sigma_gal'], bd['z'], bd['sigma_inst']) 

        plt.figure()
        plt.plot(x_fake/(1+best_z), y_fake)
        #plt.show()

        spectrum = y_fake
        
        # Looping over the number of runs specified
        for curr_loop in range(runs):
            print("working with " + str(cube_id) + " and index " + str(curr_loop))
            # Perturb the fake flux data by adding an amount of Gaussian noise
            random_noise = np.random.normal(0, data_std, len(x_fake))

            if (curr_loop < int(3/4 * runs)):
                random_noise = random_noise
            else:
                random_noise = 10 * random_noise
            
            spectrum = spectrum + random_noise

            xf_dr = x_fake / (1+best_z)

            # Want to attempt a refitting of the Gaussian doublet over the new data
            gauss_params = Parameters()
            gauss_params.add('c', value=bd['c'])
            gauss_params.add('i1', value=bd['i1'], min=0.0)
            gauss_params.add('r', value=1.3, min=0.5, max=1.5)
            gauss_params.add('i2', expr='i1/r', min=0.0)
            gauss_params.add('sigma_gal', value=bd['sigma_gal'])
            gauss_params.add('z', value=bd['z'])
            gauss_params.add('sigma_inst', value=bd['sigma_inst'], vary=False)

            gauss_model = Model(spectra_data.f_doublet)
            gauss_result = gauss_model.fit(spectrum, x=x_fake, params=gauss_params)

            new_best_fit = gauss_result.best_fit

            new_best_values = gauss_result.best_values
            new_best_sigma = new_best_values['sigma_gal']

            sigma_ratio = np.abs(best_sigma-new_best_sigma) / best_sigma
            
            ndr_mask = ((xf_dr > 3500) & (xf_dr < 3700))
        
            new_x_masked = xf_dr[ndr_mask]
            new_y_masked = spectrum[ndr_mask]

            new_signal = np.median(new_y_masked)
            new_noise = np.std(new_y_masked)

            new_sn = new_signal / new_noise

            data[i_cube][curr_loop][0] = new_signal # new signal
            data[i_cube][curr_loop][1] = new_best_sigma # new doublet sigma
            data[i_cube][curr_loop][2] = np.abs(sigma_ratio) # new fractional error
            data[i_cube][curr_loop][3] = new_sn # new S/N error
 
            plt.figure() 
            plt.plot(xf_dr, y_fake, linewidth=0.5, color="#8bc34a")
            plt.plot(xf_dr, spectrum, linewidth=0.5, color="#000000")

            plt.plot(xf_dr, new_best_fit, linewidth=0.5, color="#d32f2f")

            plt.xlabel(r'\textbf{S/N}', fontsize=15)
            plt.ylabel(r'\textbf{Flux}', fontsize=15)

            plt.tight_layout()

            uncert_lmfit_dir = "uncert_lmfit/cube_"+str(cube_id)
            if not os.path.exists(uncert_lmfit_dir):
                os.mkdir(uncert_lmfit_dir)
            plt.savefig(uncert_lmfit_dir + "/cube_"+str(cube_id)+ 
                    "_" + str(curr_loop) + ".pdf")
            plt.close("all")

            np.save("uncert_lmfit/cube_"+str(cube_id)+"/cube_"+str(cube_id)+
                    "_lmfit_perts", data[:,i_cube][:][:])

    np.save("data/lmfit_uncert_data", data)

def lmfit_graphs():
    data = np.load("data/lmfit_uncert_data.npy")
    
    def frac_error_vs_sn():
        # Fractional error vs. the signal to noise
        total_bins = 400

        #X_sn = data[:,:,3]
        X_sn = data[3][:,3]
        
        colours = spectra_data.colour_list()

        bins = np.linspace(X_sn.min(), X_sn.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(X_sn,bins)

        plt.figure()
        for i in range(len(data[:])):
            plt.scatter(data[i][:,3], data[i][:,2], c=colours[i], s=10, alpha=0.2)
        
        #plt.scatter(data[3][:,3], data[3][:,2], c="#8e24aa", s=10, alpha=0.2)

        # Running median for data
        #Y_fe = data[:,:,2] # fractional error
        Y_fe = data[3][:,2]
        running_median = [np.median(Y_fe[idx==k]) for k in range(total_bins)] 

        rm_frac_error = np.array(running_median)        
        sn_data = (bins-delta/2)
        plt.scatter(sn_data, rm_frac_error, c="#000000", s=10, alpha=0.7)

        idx = np.isfinite(rm_frac_error) # mask to mask out finite values

        fitted_poly = np.poly1d(np.polyfit(sn_data[idx], rm_frac_error[idx], 3))
        t = np.linspace(np.min(sn_data[idx]), np.max(sn_data[idx]), 200)
        plt.plot(t, fitted_poly(t), c="#d32f2f", lw=1.5, alpha=0.8) 

        plt.tick_params(labelsize=15)
        plt.xlabel(r'\textbf{S/N}', fontsize=15)
        plt.ylabel(r'\textbf{${\Delta \sigma}/{\sigma_{best}}$}', fontsize=15)

        plt.tight_layout()
        plt.savefig("uncert_lmfit/frac_error_vs_sn.pdf")
        plt.close("all")

    frac_error_vs_sn()


cubes = np.array([1804, 765, 5, 1, 767, 1578, 414, 1129, 286, 540])
#cubes = np.array([1804])

#ppxf_uncertainty(cubes, 300)
ppxf_graphs()

#lmfit_uncertainty(cubes, 300)
#lmfit_graphs()
