import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.stats import norm
from tqdm import tqdm
import os

import pywt
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import ScalarFormatter, FixedFormatter, NullFormatter
from math import log2, ceil

from Objects.SyntheticData import SyntheticData
from Objects.RollingAverage import RollAverage
from Objects.FLH import FLH
from Objects.IFLH import IFLH
from Objects.TFLH import TFLH
from Objects.ARROWS import ARROWS
from Objects.RollingAverage import RollAverage

def generate_data(
        n : int,
        segment_size : int,
        theta_type = 'hard_shift'
    ):
    segment_size = 20

    data = SyntheticData(
        n = n, 
        parameter_scale = 1, 
        sample_mean = 0, 
        sample_var = 1,
        theta_type = theta_type,
        segment_size = segment_size
    )
    data.run()

    return data

def averaging(
        N : int,
        n : int,
        theta_type : str
    ):

    SIGMA = np.sqrt(1)   #standard deviation of the noise
    BETA = 24+1e-10    #ARROWS parameter (must be > 24)
    ALPHA = 0.5/(np.log(n) ** 1)
    ALPHA_T = 1/(np.log(n) ** 1) #learning rate for FLH
    lookback = 10 #lookback
    gamma = 1.1 #\gamma > 1
    M = int(np.log(n) ** gamma) # number of bins for data thinning

    loss_flh_dep_all = np.zeros((N, n))
    loss_flh_ind_all = np.zeros((N, n))

    loss_tflh_dep_all = np.zeros((N, n))
    loss_tflh_ind_all = np.zeros((N, n))

    loss_tflh_avg_dep_all = np.zeros((N, n))
    loss_tflh_avg_ind_all = np.zeros((N, n))

    loss_arrows_dep_all = np.zeros((N, n))
    loss_arrows_ind_all = np.zeros((N, n))

    loss_roll_dep_all = np.zeros((N, n))
    loss_roll_ind_all = np.zeros((N, n))

    data_bound = np.zeros((N, n))

    for i in range(N):

        data = generate_data(n, theta_type)

        y_data_dep = data.dep
        y_data_ind = data.ind
        theta = data.theta
        data_bound[i, :] = data.bound

        #flh (iflh for speed)
        flh_algorithm = IFLH(alpha=ALPHA)
        _, loss_dep = flh_algorithm.run(y_data_dep, theta)
        _, loss_ind = flh_algorithm.run(y_data_ind, theta)
        loss_dep = np.nancumsum(loss_dep)
        loss_ind = np.nancumsum(loss_ind)

        loss_flh_dep_all[i, :] = loss_dep
        loss_flh_ind_all[i, :] = loss_ind

        #tflh
        thinned_flh_runner = TFLH(m = M, alpha = ALPHA_T) #FIX WHICH LOSS IS BEING TAKEN
        _, _, losst_dep, losst_avg_dep = thinned_flh_runner.run(y_data_dep, theta)
        _, _, losst_ind, losst_avg_ind = thinned_flh_runner.run(y_data_ind, theta)

        loss_tflh_dep_all[i, :] = losst_dep
        loss_tflh_ind_all[i, :] = losst_ind
        loss_tflh_avg_dep_all[i, :] = losst_avg_dep
        loss_tflh_avg_ind_all[i, :] = losst_avg_ind

        #arrows
        arrows_detector = ARROWS(n=n, sigma=SIGMA, beta=BETA)
        _, lossa_dep = arrows_detector.run(y_data_dep, theta)
        _, lossa_ind = arrows_detector.run(y_data_ind, theta)

        loss_arrows_dep_all[i, :] = lossa_dep
        loss_arrows_ind_all[i, :] = lossa_ind

        #rolling average
        rolling_average = RollAverage(lookback = lookback)
        _, loss_dep_roll = rolling_average.run(y_data_dep, theta)
        _, loss_ind_roll = rolling_average.run(y_data_ind, theta)
        loss_dep_roll, loss_ind_roll = np.nancumsum(loss_dep_roll), np.nancumsum(loss_ind_roll)

        loss_roll_dep_all[i, :] = loss_dep_roll
        loss_roll_ind_all[i, :] = loss_ind_roll


    avg_loss_flh_dep = np.nanmean(loss_flh_dep_all, axis=0)
    avg_loss_flh_ind = np.nanmean(loss_flh_ind_all, axis=0)

    avg_loss_tflh_dep = np.nanmean(loss_tflh_dep_all, axis=0)
    avg_loss_tflh_ind = np.nanmean(loss_tflh_ind_all, axis=0)

    avg_loss_tflh_avg_dep = np.nanmean(loss_tflh_avg_dep_all, axis=0)
    avg_loss_tflh_avg_ind = np.nanmean(loss_tflh_avg_ind_all, axis=0)

    avg_loss_arrows_dep = np.nanmean(loss_arrows_dep_all, axis=0)
    avg_loss_arrows_ind = np.nanmean(loss_arrows_ind_all, axis=0)

    avg_loss_roll_dep = np.nanmean(loss_roll_dep_all, axis=0)
    avg_loss_roll_ind = np.nanmean(loss_roll_ind_all, axis=0)

    avg_data_bound = np.nanmean(data_bound, axis=0)

    final_losses = {
        'flh_dep': avg_loss_flh_dep,
        'flh_ind': avg_loss_flh_ind,
        'tflh_dep': avg_loss_tflh_dep,
        'tflh_ind': avg_loss_tflh_ind,
        'tflh_avg_dep' : avg_loss_tflh_avg_dep,
        'tflh_avg_ind' : avg_loss_tflh_avg_ind,
        'arrows_dep': avg_loss_arrows_dep,
        'arrows_ind': avg_loss_arrows_ind,
        'roll_dep': avg_loss_roll_dep,
        'roll_ind': avg_loss_roll_ind,
        'data_bound' : avg_data_bound
    }

    return final_losses
    



if __name__ == '__main__':
    n = 250
    N = 10

    theta_type = 'hard_shift' #choose between 'hard_shift', 'hard_shift_2', or 'soft_shift'

    fin = averaging(N = N, n = n, theta_type = theta_type)

    save_dir = "results"
    os.makedirs(save_dir, exist_ok = True)

    np.savez(
        f"{save_dir}/results_{theta_type}_n{n}_N{N}.npz",
        **fin
    )
    
    
    check_flh, check_tflh, check_tflh_avg, check_arrows, check_roll = True, True, True, True, True
    check_ind, check_dep = False, True
    check_optimal = True

    start_index = 10
    x_values = np.arange(start_index, n)

    fig, ax = plt.subplots()

    if check_flh:
        if check_dep:
            ax.plot(x_values, fin['flh_dep'][start_index:], color = 'r', label = 'FLH', linewidth = 1)
        if check_ind:
            ax.plot(x_values, fin['flh_ind'][start_index:], color = 'r', label = 'FLH', linewidth = 1)
    if check_tflh:
        if check_dep:
            ax.plot(x_values, fin['tflh_dep'][start_index:], color = 'g', label = 'FLH_T')
        if check_ind:
            ax.plot(x_values, fin['tflh_ind'][start_index:], color = 'g', label = 'FLH_T')
    if check_tflh_avg:
        if check_dep:
            ax.plot(x_values, fin['tflh_avg_dep'][start_index:], color = 'orange', label = 'FLH_T Average')
        if check_ind:
            ax.plot(x_values, fin['tflh_avg_ind'][start_index:], color = 'orange', label = 'FLH_T Average')
    if check_arrows:
        if check_dep:
            ax.plot(x_values, fin['arrows_dep'][start_index:], color = 'b', label = 'ARROWS')
        if check_ind:
            ax.plot(x_values, fin['arrows_ind'][start_index:], color = 'b', label = 'ARROWS')
    if check_roll:
        if check_dep:
            ax.plot(x_values, fin['roll_dep'][start_index:], color = 'purple', label = "Rolling Average")
        if check_ind:
            ax.plot(x_values, fin['roll_ind'][start_index:], color = 'purple', label = "Rolling Average")
    if check_optimal: #dashed optimal line, play around with constant to get good plot
        ax.plot(x_values, (40*fin['data_bound'])[start_index:], color = 'k', label = r'$C_n^{2/3}n^{1/3}$', linestyle = 'dashed')


    ax.grid(True, which="major", linestyle='--', linewidth=0.5)
    ax.legend()
    if check_ind:
        ax.set_title("Independent Errors")
    if check_dep:
        ax.set_title("Dependent Errors")
    

    filename = f"{save_dir}/plot_{theta_type}_n{n}_N{N}.png"
    plt.savefig(filename, dpi = 300, bbox_inches = "tight")

    plt.show()
    

    

        

        