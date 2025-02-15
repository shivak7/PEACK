import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def movements(ts,peak_prom, axis=1, dist = None):
    pidx = []

    if(len(ts.shape)<2):
        ts = np.reshape(ts,(-1,1))

    Ndims = ts.shape[axis]

    for i in range(Ndims):
        idx,_ = find_peaks(ts[:,i],distance=dist, prominence= peak_prom/3, plateau_size=[1,15])#, rel_height=peak_prom) #,prominence=None, plateau_size=[0,15])
        pidx.append(idx)
    #     plt.subplot(Ndims,1,i+1)
    #     plt.plot(ts[:,i])
    #     plt.plot(np.arange(0,len(ts[:,i]))[idx], ts[idx,i],'x')
    # plt.show(block=True)

    start_indices = pidx[0][::2]
    stop_indices = pidx[0][1::2]



    if(len(np.squeeze(pidx))%2)==1:                     #len will be odd if there are more start then stop indices, therefore delete the last start index
        start_indices = start_indices[:-1]


    # for s in range(len(start_indices)):
    #     for i in range(Ndims):
    #         trial = ts[start_indices[s]:stop_indices[s],i]
    #         plt.plot(trial)
    # plt.show()
    #import pdb; pdb.set_trace()

    #r = stop_indices - start_indices
    #print(np.amin(r), '\t', np.amax(r))
    #import pdb; pdb.set_trace()
    return np.array(start_indices), np.array(stop_indices)
    #raise SystemExit


    #return Trials

class split_by_repititions:

    def __init__(self):
        pass

    def using_min_speed(ts, plat_size=[1,15], distance=None, show_plots=False):

        Ndims = len(ts.shape)
        ts_speed = np.linalg.norm(np.diff(ts,axis=0),axis=-1)
        ts_speed = np.append(ts_speed[0], ts_speed)
        prominence = 2*np.std(ts_speed)
        idx,_ = find_peaks(ts_speed, distance=distance, prominence=prominence, plateau_size=plat_size)

        min_idx = []
        #min_idx.append(0)
        for i in range(0,len(idx)-1):

            sub_ts = ts_speed[idx[i]:idx[i+1]]
            m_sub_idx = idx[i] + np.argmin(sub_ts)
            min_idx.append(m_sub_idx)
        #min_idx.append(len(ts) - 1)

        x_range = np.arange(0, len(ts_speed), 1)

        #import pdb; pdb.set_trace()

        if show_plots:
            plt.plot(x_range, ts_speed)
            plt.plot(x_range[idx], ts_speed[idx], 'k*')
            plt.plot(x_range[min_idx], ts_speed[min_idx], 'r*')
            plt.show()

        return np.array(min_idx)
        


        
        


def trials(ts, start_indices, stop_indices):

    Trials = []
    #ts = ts.reshape(-1,1)
    for i in range(len(start_indices)):
        trial = ts[start_indices[i]:stop_indices[i], :]
        Trials.append(trial)

    return Trials
