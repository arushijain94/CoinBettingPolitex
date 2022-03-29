from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
matplotlib.style.use('classic')
plt.rcParams.update({'font.size': 15})

def rolling_mean_along_axis(data, W, axis=-1):
    """
    plotting data with rolling mean across last axis
    """
    # data : =[nruns X niterations]
    # W : Window size
    # axis : Axis along which we will apply rolling/sliding mean
    hW = W // 2
    L = data.shape[axis] - W + 1
    indexer = [slice(None) for _ in range(data.ndim)]
    indexer[axis] = slice(hW, hW + L)
    new_data = uniform_filter1d(data, W, axis=axis)[tuple(indexer)]
    # print("new data shape:", new_data.shape)
    nruns = new_data.shape[0]
    niterations = new_data.shape[1]
    ci = 1.96 * np.std(new_data, axis=0) / np.sqrt(nruns)  # 95% confidence interval
    mean_val = np.mean(new_data, axis=0)
    return mean_val, ci, niterations

