import sigma_uncertainties
import spectra_data

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r'\boldmath']

from lmfit import Parameters, Model

def ppxf_graphs():
    data = np.load("data/ppxf_uncert_data.npy")

    colours = spectra_data.colour_list()

    def sn_vs_frac_error():
        # S/N vs. fractional error
        total_bins2 = 400
        X_sigma = data[:,:,3]
        
        bins = np.linspace(X_sigma.min(), X_sigma.max(), total_bins2)
        delta = bins[1]-bins[0]
        idx  = np.digitize(X_sigma,bins)

        plt.figure()
        for i in range(len(data[:])):
            plt.scatter(data[i][:,3], data[i][:,5], c=colours[i], s=15, alpha=0.2)

        # Running median for data
        Y_sn = data[:,:,5]
        running_median3 = [np.median(Y_sn[idx==k]) for k in range(total_bins2)]

        rm3 = np.array(running_median3)
        xd = bins-delta/2

        #plt.plot(xd, rm3, c="#000000", lw=1.5, alpha=0.7)
        plt.scatter(xd, rm3, c="#000000", s=15, alpha=0.7)

        # Fitting an a/x line to the data 
        curve_params = Parameters()
        curve_params.add('a', value=1)
        curve_model = Model(sigma_uncertainties.curve)

        idx = np.isfinite(rm3) # mask to mask out finite values
        curve_result = curve_model.fit(rm3[idx], x=xd[idx], params=curve_params)
    
        curve_bf = curve_result.best_fit
        curve_bp = curve_result.best_values

        np.save("uncert_ppxf/vel_curve_best_values_ppxf", np.array([curve_bp['a']]))     
        
        plt.plot(xd[idx], curve_bf, c="#d32f2f", lw=2, alpha=1.0)
       
        plt.xlim([0,10])
        plt.ylim([10**(-4.4),10**(-2.9)])
        plt.yscale('log')
        plt.tick_params(labelsize=20)
        plt.xlabel(r'\textbf{S/N}', fontsize=20)
        plt.ylabel(r'\textbf{Stellar ${|\Delta V|}/{V_{best}}$}', fontsize=20)

        plt.tight_layout()
        plt.savefig("uncert_ppxf/ppxf_vel_frac_error_vs_sn.pdf",bbox_inches="tight")
        plt.close("all")

    sn_vs_frac_error()

def lmfit_graphs():
    data = np.load("data/lmfit_uncert_data.npy")
    
    def frac_error_vs_sn():
        # Fractional error vs. the signal to noise
        total_bins = 400

        X_sn = data[:,:,3]
        
        colours = spectra_data.colour_list()

        bins = np.linspace(X_sn.min(), X_sn.max(), total_bins)
        delta = bins[1]-bins[0]
        idx  = np.digitize(X_sn,bins)

        plt.figure()
        for i in range(len(data[:])):
            plt.scatter(data[i][:,3], data[i][:,6], c=colours[i], s=15, alpha=0.2)
        
        # Running median for data
        Y_fe = data[:,:,6] # fractional error
        running_median = [np.median(Y_fe[idx==k]) for k in range(total_bins)] 

        rm_frac_error = np.array(running_median)        
        sn_data = (bins-delta/2)
        plt.scatter(sn_data, rm_frac_error, c="#000000", s=15, alpha=0.7)

        # Fitting an a/x line to the data 
        curve_params = Parameters()
        curve_params.add('a', value=1)
        curve_model = Model(sigma_uncertainties.curve)

        idx = np.isfinite(rm_frac_error) # mask to mask out finite values
        curve_result = curve_model.fit(rm_frac_error[idx], x=sn_data[idx], 
                params=curve_params)
    
        curve_bf = curve_result.best_fit
        curve_bp = curve_result.best_values
        np.save("uncert_lmfit/vel_curve_best_values_lmfit", 
                np.array([curve_bp['a']]))

        gen_xd = np.linspace(0,150, 500)
        gen_yd = sigma_uncertainties.curve(gen_xd, curve_bp['a'])
        plt.plot(gen_xd, gen_yd, c="#d32f2f", lw=2.0, alpha=1.0)

        plt.xlim([0,10])
        plt.ylim([10**(-4),10**(-2.3)])
        plt.yscale('log')
        plt.tick_params(labelsize=20)
        plt.xlabel(r'\textbf{S/N}', fontsize=20)
        plt.ylabel(r'\textbf{[OII] ${|\Delta V|}/{V_{best}}$}', fontsize=20)

        plt.tight_layout()
        plt.savefig("uncert_lmfit/lmfit_vel_frac_error_vs_sn.pdf",bbox_inches="tight")
        plt.close("all")

    frac_error_vs_sn()

if __name__ == '__main__':
    ppxf_graphs()
    lmfit_graphs()
