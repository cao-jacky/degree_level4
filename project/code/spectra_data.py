import numpy as np

""" Functions which are normally used multiple times, easier to call them from a 
    predefined placed. """

def lmfit_data(cube_id):
    # parameters from lmfit
    cube_result_file = ("cube_results/cube_" + str(cube_id) + "/cube_" + str(cube_id) 
            + "_lmfit.txt")
    cube_result_file = open(cube_result_file)

    line_count = 0 
    for crf_line in cube_result_file:
        if (line_count == 15):
            curr_line = crf_line.split()
            c = float(curr_line[1])
        if (line_count == 16):
            curr_line = crf_line.split()
            i1 = float(curr_line[1])
        if (line_count == 18):
            curr_line = crf_line.split()
            i2 = float(curr_line[1])
        if (line_count == 19):
            curr_line = crf_line.split()
            sigma_gal = float(curr_line[1])
        if (line_count == 20):
            curr_line = crf_line.split()
            z = float(curr_line[1])
            err_z = float(curr_line[3])
        if (line_count == 21):
            curr_line = crf_line.split()
            sigma_inst = float(curr_line[1])
        line_count += 1

    return {'c': c, 'i1': i1, 'i2': i2, 'sigma_gal': sigma_gal, 'z': z, 'err_z': err_z,
            'sigma_inst': sigma_inst}

def spectral_lines():
    sl = {
        'emis': {
            '':             '3727.092', 
            'OII':          '3728.875',
            'HeI':          '3889.0',
            'SII':          '4072.3',
            'H$\delta$':    '4101.89',
            'H$\gamma$':    '4341.68'
            },
        'abs': {
            r'H$\theta$':   '3798.976',
            'H$\eta$':      '3836.47',
            'CaK':          '3934.777',
            'CaH':          '3969.588',
            'G':            '4305.61',
            'Mg':           '5176.7',
            },
        'iron': {
            'FeI1':     '4132.0581',
            'FeI2':     '4143.8682',
            'FeI3':     '4202.0293', 
            'FeI4':     '4216.1836',
            'FeI5':     '4250.7871',
            'FeI6':     '4260.4746',
            'FeI7':     '4271.7607',
            'FeI8':     '4282.4028',
            }
        }
    return sl

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

def colour_list():
    colours = [
            "#f44336",
            "#d81b60",
            "#8e24aa",
            "#5e35b1",
            "#3949ab",
            "#1e88e5",
            "#0097a7",
            "#43a047",
            "#fbc02d",
            "#616161"
            ]
    return colours
