import datetime

def analysis_complete(data_dir, stk_f_n, gss_result, init_pms, opti_pms, 
            sn_line_csqs, sn_gauss_csqs, signal_noise, sn_line_bpms, sn_line_data,
            sn_gauss_bpms, sn_gauss_data):

    lmfit_file = open(data_dir + '/' + stk_f_n + '_lmfit.txt', 'w') 
    lmfit_file.write("Analysis performed on " + str(datetime.datetime.now()) + "\n\n")
    lmfit_file.write("Output from lmfit is the following: \n")
    lmfit_file.write(gss_result.fit_report()) 
    
    data_file = open(data_dir + '/' + stk_f_n + '_fitting.txt', 'w')
    data_file.write("Results produced on " + str(datetime.datetime.now()) + "\n\n")

    data_file.write("# Model fitting for [OII] Gaussian Doublet \n")
    data_file.write("## Initial Parameters \n")
    data_file.write("'c': " + str(init_pms['c'])  + "\n")
    data_file.write("'i1': " + str(init_pms['i1'])  + "\n")
    data_file.write("'i2': " + str(init_pms['i2'])  + "\n")
    data_file.write("'sigma1': " + str(init_pms['sigma1'])  + "\n")
    data_file.write("'z': " + str(init_pms['z'])  + "\n\n")

    data_file.write("## Optimal Parameters \n")
    data_file.write("'c': " + str(opti_pms['c'])  + "\n")
    data_file.write("'i1': " + str(opti_pms['i1'])  + "\n")
    data_file.write("'i2': " + str(opti_pms['i2'])  + "\n")
    data_file.write("'sigma1': " + str(opti_pms['sigma1'])  + "\n")
    data_file.write("'z': " + str(opti_pms['z'])  + "\n\n")

    data_file.write("# Calculating the signal-to-noise from doublet region \n")
    data_file.write("Fitting a straight line to the data with parameter: \n")
    data_file.write("'c': " + str(sn_line_bpms['c']) + "\n\n")
    data_file.write("For this model, the following chi-squared and reduced " 
        + "chi-squared values were produced: " + str(sn_line_csqs['chisq']) + " and " 
        + str(sn_line_csqs['chisq_red']) + " \n\n")

    data_file.write("Fitting a gaussian to the data with parameters: \n")
    data_file.write("'c': " + str(sn_gauss_bpms['c']) + "\n")
    data_file.write("'i1': " + str(sn_gauss_bpms['i1']) + "\n")
    data_file.write("'mu': " + str(sn_gauss_bpms['mu']) + "\n")
    data_file.write("'sigma1': " + str(sn_gauss_bpms['sigma1']) + "\n\n")    
    data_file.write("For this model, the following chi-squared and reduced " 
        + "chi-squared values were produced: " + str(sn_gauss_csqs['chisq']) + " and " 
        + str(sn_gauss_csqs['chisq_red']) + " \n\n")

    data_file.write("Using the two full values of chi-squared we can calculate a" 
            + " signal to noise value of: " + str(signal_noise) + "\n\n")
