import matplotlib.pyplot as plt
import numpy as np

import cube_analysis

def model_data_overlay(cube_id):
    
    x_model_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_x.npy")  
    y_model = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_model.npy")
    print(x_model_data)
    print(y_model)

    x_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbd_x.npy") 
    y_data = np.load("cube_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_cbs_y.npy")

    #print(np.shape(x_data), np.shape(y_data), np.shape(y_model))
    
    plt.figure()
    plt.plot(x_model_data, y_model, linewidth=0.5, color="#000000")
    #plt.plot(x_data, y_data, linewidth=0.5, color="#42a5f5")
    plt.savefig("ppxf_results/cube_" + str(int(cube_id)) + "data_model.pdf")

def chi_squared_cal(cube_id):
    # defining wavelength as the x-axis
    x_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_x.npy")

    # defining the flux from the data and model
    y_data = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_flux.npy")
    y_model = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_model.npy")

    # scale down y data 
    y_data = y_data/np.median(y_data)
    
    # noise spectra will be used as in the chi-squared calculation
    noise = np.load("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + 
            str(int(cube_id)) + "_noise.npy")

    residual = y_data - y_model
    res_median = np.median(residual)
    res_stddev = np.std(residual)

    print(noise)
    print(residual)

    mask = ((residual < np.std(residual)) & (residual > -np.std(residual))) 
 
    chi_sq = (y_data[mask] - y_model[mask])**2 / noise[mask]**2
    total_chi_sq = np.sum(chi_sq)

    total_points = len(noise)
    reduced_chi_sq = total_chi_sq / total_points
    print(reduced_chi_sq) 

    plt.figure()

    plt.plot(x_data, y_data, linewidth=0.5, color="#000000")
    plt.plot(x_data, y_model, linewidth=0.5, color="#b71c1c")
    plt.plot(x_data, noise, linewidth=0.5, color="#8e24aa")

    plt.axhline(res_stddev, linewidth=0.5, color="#000000", alpha=0.3)
    plt.axhline(-res_stddev, linewidth=0.5, color="#000000", alpha=0.3)

    plt.scatter(x_data, residual, s=3, color="#43a047")
    plt.savefig("ppxf_results/cube_" + str(int(cube_id)) + "/cube_" + str(int(cube_id))
            + "_fitted.pdf")


chi_squared_cal(5)

#model_data_overlay(549)
