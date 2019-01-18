import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import cube_reader
import ppxf_fitter_kinematics_sdss

from astropy.io import fits

from lmfit import Parameters, Model
from lmfit.models import VoigtModel, ConstantModel

import spectra_data

def diag_results(cube_id):
    def f_doublet(x, c, i1, i2, sigma_gal, z, sigma_inst):
        """ function for Gaussian doublet """  
        dblt_mu = [3727.092, 3729.875] # the actual non-redshifted wavelengths
        l1 = dblt_mu[0] * (1+z)
        l2 = dblt_mu[1] * (1+z)

        sigma = np.sqrt(sigma_gal**2 + sigma_inst**2)

        norm = (sigma*np.sqrt(2*np.pi))
        term1 = ( i1 / norm ) * np.exp(-(x-l1)**2/(2*sigma**2))
        term2 = ( i2 / norm ) * np.exp(-(x-l2)**2/(2*sigma**2)) 
        return (c*x + term1 + term2)

    with PdfPages('diagnostics/cube_'+str(cube_id)+'_diagnostic.pdf') as pdf: 
        analysis = cube_reader.analysis("/Volumes/Jacky_Cao/University/level4/" + 
                "project/cubes_better/cube_"+str(cube_id)+".fits", 
                "data/skyvariance_csub.fits") 

        # calling data into variables
        icd = analysis['image_data']
        
        segd = analysis['spectra_data']['segmentation']

        sr = analysis['sr']
        df_data = analysis['df_data']
        gs_data = analysis['gs_data']
        snw_data = analysis['snw_data']

        # images of the galaxy
        f, (ax1, ax2)  = plt.subplots(1, 2)
        ax1.imshow(icd['median'], cmap='gray_r') 
        ax1.set_title(r'\textbf{Galaxy Image: Median}', fontsize=13)    
        ax1.set_xlabel(r'\textbf{Pixels}', fontsize=13)
        ax1.set_ylabel(r'\textbf{Pixels}', fontsize=13) 

        ax2.imshow(icd['sum'], cmap='gray_r')
        ax2.set_title(r'\textbf{Galaxy Image: Sum}', fontsize=13)        
        ax2.set_xlabel(r'\textbf{Pixels}', fontsize=13)
        ax2.set_ylabel(r'\textbf{Pixels}', fontsize=13)    
        f.subplots_adjust(wspace=0.4)
     
        pdf.savefig()
        plt.close()
        
        # ---------------------------------------------------------------------- #
    
        # segmentation area used to extract the 1D spectra 
        segd_mask = ((segd == cube_id))
        
        plt.figure()
        plt.title(r'\textbf{Segmentation area used to extract 1D spectra}', 
                fontsize=13)
        plt.imshow(np.rot90(segd_mask, 1), cmap='Paired')
        plt.xlabel(r'\textbf{Pixels}', fontsize=13)
        plt.ylabel(r'\textbf{Pixels}', fontsize=13)
        pdf.savefig()
        plt.close()

        # ---------------------------------------------------------------------- #

        # spectra plotting 
        f, (ax1, ax2)  = plt.subplots(2, 1) 
        # --- redshifted data plotting
        cbd_x   = np.linspace(sr['begin'], sr['end'], sr['steps'])
         
        ## plotting our cube data
        cbs_y   = gs_data['gd_shifted'] 
        ax1.plot(cbd_x, cbs_y, linewidth=0.5, color="#000000")

        ## plotting our sky noise data
        snd_y   = snw_data['sky_regions'][:,1]
        ax1.plot(cbd_x, snd_y, linewidth=0.5, color="#f44336", alpha=0.5)

        ## plotting our [OII] region
        ot_x    = df_data['x_region']
        ot_y    = df_data['y_region']
        ax1.plot(ot_x, ot_y, linewidth=0.5, color="#00c853") 

        ## plotting the standard deviation region in the [OII] section
        std_x   = df_data['std_x']
        std_y   = df_data['std_y']
        ax1.plot(std_x, std_y, linewidth=0.5, color="#00acc1") 
        
        pu_lines = gs_data['pu_peaks']
        for i in range(len(pu_lines)):
            srb = sr['begin']
            ax1.axvline(x=(pu_lines[i]), linewidth=0.5, color="#ec407a", alpha=0.2)

        ax1.set_title(r'\textbf{Spectra: cross-section redshifted}', fontsize=13)    
        ax1.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
        ax1.set_ylabel(r'\textbf{Flux}', fontsize=13)
        ax1.set_ylim([-1000,5000]) # setting manual limits for now

        # --- corrected redshift
        crs_x   = np.linspace(sr['begin'], sr['end'], sr['steps'])
        rdst    = gs_data['redshift']

        sp_lines = gs_data['spectra']

        ## corrected wavelengths
        corr_x  = crs_x / (1+rdst)

        ## plotting our cube data
        cps_y   = gs_data['gd_shifted']
        ax2.plot(corr_x, cps_y, linewidth=0.5, color="#000000")

        ## plotting our sky noise data
        sn_y    =  gs_data['sky_noise']
        ax2.plot(corr_x, sn_y, linewidth=0.5, color="#e53935")

        ## plotting spectra lines
        for e_key, e_val in sp_lines['emis'].items():
            spec_line = float(e_val)
            ax2.axvline(x=spec_line, linewidth=0.5, color="#00c853")
            ax2.text(spec_line-10, 4800, e_key, rotation=-90)

        ax2.set_title(r'\textbf{Spectra: cross-section corrected}', fontsize=13)        
        ax2.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
        ax2.set_ylabel(r'\textbf{Flux}', fontsize=13)
        ax2.set_ylim([-500,5000]) # setting manual limits for now

        f.subplots_adjust(hspace=0.5)
        pdf.savefig()
        plt.close()

        # ---------------------------------------------------------------------- #

        # OII doublet region
        ot_fig  = plt.figure()
        # plotting the data for the cutout [OII] region
        ot_x    = df_data['x_region']
        ot_y    = df_data['y_region']
        plt.plot(ot_x, ot_y, linewidth=0.5, color="#000000")

        ## plotting the standard deviation region in the [OII] section
        std_x   = df_data['std_x']
        std_y   = df_data['std_y']
        plt.plot(std_x, std_y, linewidth=0.5, color="#00acc1") 

        dblt_rng    = df_data['doublet_range']
        ot_x_b, ot_x_e  = dblt_rng[0], dblt_rng[-1]
        x_ax_vals   = np.linspace(ot_x_b, ot_x_e, 1000)

        # lmfit 
        lm_init     = df_data['lm_init_fit']
        lm_best     = df_data['lm_best_fit'] 

        plt.plot(ot_x, lm_best, linewidth=0.5, color="#1e88e5")
        plt.plot(ot_x, lm_init, linewidth=0.5, color="#43a047", alpha=0.5)
        
        lm_params   = df_data['lm_best_param']
        lm_params   = [prm_value for prm_key, prm_value in lm_params.items()]
        c, i_val1, i_val2, sig_g, rdsh, sig_i = lm_params

        dblt_mu = [3727.092, 3729.875] # the actual non-redshifted wavelengths for OII
        l1 = dblt_mu[0] * (1+rdsh)
        l2 = dblt_mu[1] * (1+rdsh)
        
        sig = np.sqrt(sig_g**2 + sig_i**2) 
        norm = (sig*np.sqrt(2*np.pi))

        lm_y1 = c + ( i_val1 / norm ) * np.exp(-(ot_x-l1)**2/(2*sig**2))
        lm_y2 = c + ( i_val2 / norm ) * np.exp(-(ot_x-l2)**2/(2*sig**2))

        plt.plot(ot_x, lm_y1, linewidth=0.5, color="#e64a19", alpha=0.7) 
        plt.plot(ot_x, lm_y2, linewidth=0.5, color="#1a237e", alpha=0.7)

        # plotting signal-to-noise straight line and gaussian to verify it works
        sn_line     = df_data['sn_line']
        sn_gauss    = df_data['sn_gauss']

        plt.title(r'\textbf{OII doublet region}', fontsize=13)        
        plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
        plt.ylabel(r'\textbf{Flux}', fontsize=13)
        plt.ylim([-500,5000]) # setting manual limits for now
        pdf.savefig()
        plt.close()

        # ---------------------------------------------------------------------- #

        # plotting pPXF data
        # defining wavelength as the x-axis
        x_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_lamgal.npy")

        # defining the flux from the data and model
        y_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_flux.npy")
        y_model = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_model.npy")

        # scaled down y data 
        y_data_scaled = y_data/np.median(y_data)

        # opening cube to obtain the segmentation data
        cube_file = ("/Volumes/Jacky_Cao/University/level4/project/cubes_better/cube_"
            + str(cube_id) + ".fits")
        hdu = fits.open(cube_file)
        segmentation_data = hdu[2].data
        seg_loc_rows, seg_loc_cols = np.where(segmentation_data == cube_id)
        signal_pixels = len(seg_loc_rows) 

        # noise spectra will be used as in the chi-squared calculation
        noise = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_noise.npy")
        noise_median = np.median(noise)
        noise_stddev = np.std(noise) 

        residual = y_data_scaled - y_model
        res_median = np.median(residual)
        res_stddev = np.std(residual)

        noise = noise
        
        mask = ((residual < res_stddev) & (residual > -res_stddev)) 
     
        chi_sq = (y_data_scaled[mask] - y_model[mask])**2 / noise[mask]**2
        total_chi_sq = np.sum(chi_sq)

        total_points = len(chi_sq)
        reduced_chi_sq = total_chi_sq / total_points

        # spectral lines
        sl = spectra_data.spectral_lines() 

        # parameters from lmfit
        lm_params = spectra_data.lmfit_data(cube_id)
        c = lm_params['c']
        i1 = lm_params['i1']
        i2 = lm_params['i2']
        sigma_gal = lm_params['sigma_gal']
        z = lm_params['z']
        sigma_inst = lm_params['sigma_inst']

        plt.figure()

        plt.plot(x_data, y_data_scaled, linewidth=1.1, color="#000000")
        plt.plot(x_data, y_data_scaled+noise_stddev, linewidth=0.1, color="#616161", 
                alpha=0.1)
        plt.plot(x_data, y_data_scaled-noise_stddev, linewidth=0.1, color="#616161", 
                alpha=0.1)
        
        # plotting over the OII doublet
        doublets = np.array([3727.092, 3728.875])
        #dblt_av = np.average(doublets) * (1+z)
        dblt_av = np.average(doublets)

        dblt_x_mask = ((x_data > dblt_av-20) & (x_data < dblt_av+20))
        doublet_x_data = x_data[dblt_x_mask]
        doublet_data = f_doublet(doublet_x_data, c, i1, i2, sigma_gal, z, sigma_inst)
        doublet_data = doublet_data / np.median(y_data)
        plt.plot(doublet_x_data, doublet_data, linewidth=0.5, color="#9c27b0")

        max_y = np.max(y_data_scaled)
        # plotting spectral lines
        for e_key, e_val in sl['emis'].items():
            spec_line = float(e_val)
            #spec_line = float(e_val) * (1+z)
            spec_label = e_key

            if (e_val in str(doublets)):
                alpha_line = 0.2
            else:
                alpha_line = 0.7
                
            alpha_text = 0.75

            plt.axvline(x=spec_line, linewidth=0.5, color="#1e88e5", alpha=alpha_line)
            plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=alpha_text,
                    weight="bold", fontsize=15) 

        for e_key, e_val in sl['abs'].items():
            spec_line = float(e_val)
            #spec_line = float(e_val) * (1+z)
            spec_label = e_key

            plt.axvline(x=spec_line, linewidth=0.5, color="#ff8f00", alpha=0.7)
            plt.text(spec_line-3, max_y, spec_label, rotation=-90, alpha=0.75,
                    weight="bold", fontsize=15)

        # iron spectral lines
        for e_key, e_val in sl['iron'].items(): 
            spec_line = float(e_val)
            #spec_line = float(e_val) * (1+z)

            plt.axvline(x=spec_line, linewidth=0.5, color="#bdbdbd", alpha=0.3)

        plt.plot(x_data, y_model, linewidth=1.5, color="#b71c1c")

        residuals_mask = (residual > res_stddev) 
        rmask = residuals_mask

        #plt.scatter(x_data[rmask], residual[rmask], s=3, color="#f44336", alpha=0.5)
        plt.scatter(x_data[mask], residual[mask]-1, s=3, color="#43a047")

        plt.tick_params(labelsize=13)
        plt.title(r'\textbf{Spectra with pPXF overlayed}', fontsize=13)
        plt.xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)
        plt.ylabel(r'\textbf{Relative Flux}', fontsize=13)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # ---------------------------------------------------------------------- #

        # Voigt fitted region
        # Running pPXF fitting routine
        best_fit = ppxf_fitter_kinematics_sdss.kinematics_sdss(cube_id, 0, "all")
        best_fit_vars = best_fit['variables']

        data_wl = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_cbd_x.npy") # 'x-data'
        data_spec = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_cbs_y.npy") # 'y-data'

        # y-data which has been reduced down by median during pPXF running
        galaxy = best_fit['y_data']

        model_wl = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_lamgal.npy") 
        model_spec = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
                str(int(cube_id)) + "_model.npy")

        # parameters from lmfit
        lm_params = spectra_data.lmfit_data(cube_id)
        z = lm_params['z']
        sigma_inst = lm_params['sigma_inst']

        # masking out the region of CaH and CaK
        calc_rgn = np.array([3900,4000]) 
        
        data_rgn = calc_rgn * (1+z)
        data_mask = ((data_wl > data_rgn[0]) & (data_wl < data_rgn[1]))
        data_wl_masked = data_wl[data_mask]
        data_spec_masked = data_spec[data_mask]

        data_spec_masked = data_spec_masked / np.median(data_spec_masked)
        
        model_rgn = calc_rgn
        model_mask = ((model_wl > calc_rgn[0]) & (model_wl < calc_rgn[1]))
        model_wl_masked = model_wl[model_mask]
        model_spec_masked = model_spec[model_mask]

        z_wl_masked = model_wl_masked * (1+z) # redshifted wavelength range
        galaxy_masked = galaxy[model_mask]

        # Applying the lmfit routine to fit two Voigt profiles over our spectra data
        vgt_pars = Parameters()
        vgt_pars.add('sigma_inst', value=sigma_inst, vary=False)
        vgt_pars.add('sigma_gal', value=1.0, min=0.0)

        vgt_pars.add('z', value=z)

        vgt_pars.add('v1_amplitude', value=-0.1, max=0.0)
        vgt_pars.add('v1_center', expr='3934.777*(1+z)')
        vgt_pars.add('v1_sigma', expr='sqrt(sigma_inst**2 + sigma_gal**2)', min=0.0)
        #vgt_pars.add('v1_gamma', value=0.01)

        vgt_pars.add('v2_amplitude', value=-0.1, max=0.0)
        vgt_pars.add('v2_center', expr='3969.588*(1+z)')
        vgt_pars.add('v2_sigma', expr='v1_sigma')
        #vgt_pars.add('v2_gamma', value=0.01) 

        vgt_pars.add('c', value=0)

        voigt = VoigtModel(prefix='v1_') + VoigtModel(prefix='v2_') + ConstantModel()

        vgt_result = voigt.fit(galaxy_masked, x=z_wl_masked, params=vgt_pars)

        opt_pars = vgt_result.best_values
        best_fit = vgt_result.best_fit

        # Plotting the spectra
        fig, ax = plt.subplots()
        ax.plot(z_wl_masked, galaxy_masked, lw=1.5, c="#000000", alpha=0.3)
        ax.plot(z_wl_masked, model_spec_masked, lw=1.5, c="#00c853")
        ax.plot(z_wl_masked, best_fit, lw=1.5, c="#e53935")

        ax.tick_params(labelsize=13)
        ax.set_ylabel(r'\textbf{Relative Flux}', fontsize=13)
        ax.set_xlabel(r'\textbf{Wavelength (\AA)}', fontsize=13)

        plt.title(r'\textbf{Voigt Fitted Region}', fontsize=15)
        fig.tight_layout()
        pdf.savefig()
        plt.close()

        # ---------------------------------------------------------------------- #
        
        # Values for diagnostics
        catalogue = np.load("data/matched_catalogue.npy")
        cat_loc = np.where(catalogue[:,0] == cube_id)[0]

        cube_data = catalogue[cat_loc][0]
        vmag = cube_data[5]

        sigma_sn_data = np.load("data/ppxf_fitter_data.npy")
        sigma_sn_loc = np.where(sigma_sn_data[:][:,0][:,0] == cube_id)[0]

        ss_indiv_data = sigma_sn_data[sigma_sn_loc][0][0]
        ssid = ss_indiv_data

        plt.figure()
        plt.title('Variables and numbers for cube ' + str(cube_id), fontsize=15)
        plt.text(0.0, 0.9, "HST V-band magnitude: " + str(vmag))
        plt.text(0.0, 0.85, "S/N from spectra: " + str(ssid[7]))

        plt.text(0.0, 0.75, "OII sigma lmfit: " + str(ssid[1]))
        plt.text(0.0, 0.7, "OII sigma pPXF: " + str(ssid[5]))

        plt.text(0.0, 0.6, "Voigt sigma lmfit: " + str(ssid[11]))
        plt.text(0.0, 0.55, "Voigt sigma pPXF: " + str(ssid[10]))

        plt.axis('off')
        pdf.savefig()
        plt.close()


        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'cube_'+str(cube_id)+' diagnostics'
        d['Author'] = u'Jacky Cao'
        #d['Subject'] = 'How to create a multipage pdf file and set its metadata'
        #d['Keywords'] = 'PdfPages multipage keywords author title subject'
        #d['CreationDate'] = datetime.datetime(2009, 11, 13)
        d['CreationDate'] = datetime.datetime.today()

#diag_results(1578)
