import numpy as np

def fractional_error():
    data = np.load("data/ppxf_fitter_data.npy")

    first_layer = data[:][:,0] # first layer of the array, the full fitted spectrum
    
    for i_fl in range(len(first_layer)):
        cube_id = int(first_layer[:,0][i_fl])
        frac_err = first_layer[:,16][i_fl]
        signal_noise = first_layer[:,7][i_fl]

        print(cube_id, frac_err, signal_noise)

    print("S/N median: " + str(np.nanmedian(first_layer[:,7])))

    fe_med = np.nanmedian(first_layer[:,16]) # median fractional error
    print(fe_med)

    ppxf_curve_params = np.load("uncert_ppxf/curve_best_values_ppxf.npy")
    ppxf_a = ppxf_curve_params
    print(ppxf_a)

    print(ppxf_a/4)

if __name__ == '__main__':
    fractional_error()
