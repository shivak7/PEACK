import numpy as np
import matplotlib.pyplot as plt
import PEACK_Filters as PF
from PEACKMetrics import metrics

def velocity(ts, fs, axis = -1):

        v = np.diff(ts,axis = axis)/(1.0/fs)
        return v

def acceleration(ts = None, vs = None, fs = 60, axis = -1): #Can calculate acceleration from position or velocity data

    if not(ts is None) and (vs is None):
        v = velocity(ts, fs, axis)
        a = np.diff(v, axis = axis)/(1.0/fs)
    elif not(vs is None) and (ts is None):
        a = np.diff(vs, axis = axis)/(1.0/fs)
    else:
        print("Error calculating acceleration! Either a position or velocity time series must be specified!")
        raise SystemError
    return a

def autocorr_delay(sig):

    acf = np.correlate(sig, sig, 'full')[-len(sig):]
    inflection = np.diff(np.sign(np.diff(acf))) # Find the second-order differences
    peaks = (inflection < 0).nonzero()[0] + 1 # Find where they are negative
    delay = peaks[acf[peaks].argmax()] # Of those, find the index with the maximum value
    return delay

def va_process(ts, fs, axis = -1):

    v = velocity(ts, fs, axis=axis)
    v = np.apply_along_axis(PF.median_filter_vector, axis, v, *[fs, 0.1])
    v = np.apply_along_axis(PF.butter_lowpass_filter, axis, v, *[3, fs, 3])
    v = np.apply_along_axis(PF.exp_smoothing_vector, axis, v, *[fs, 0.9])
    #speed = np.linalg.norm(v,2, axis=1)

    a = acceleration(None, v, fs, axis=axis)
    a = np.apply_along_axis(PF.median_filter_vector, axis, a, *[fs, 0.1])
    a = np.apply_along_axis(PF.butter_lowpass_filter, axis, a, *[3, fs, 3])
    a = np.apply_along_axis(PF.exp_smoothing_vector, axis, a, *[fs, 0.9])

    return v,a

def Eval_trajectories(trials, v_trials, a_trials, fs):

    #plt.figure()
    L = len(trials)
    v_peak = np.zeros((L,1))    #Peak speed
    vt_peak = np.zeros((L,1))   #Time-to-peak speed
    a_peak = np.zeros((L,1))    #Peak acceleration
    v_smooth = np.zeros((L,1))    #Jerk - Smoothness of velocity curve
    #traj_dev = np.zeros((L,1))    #Trajectory deviation
    max_trial_length = 0
    for i in range(L):
        speed = np.linalg.norm(v_trials[i],2, axis=1)
        inst_acc = np.linalg.norm(a_trials[i],2, axis=1)

        v_peak[i] = metrics.peak_value(speed)
        vt_peak[i] = metrics.time_to_peak(speed)
        a_peak[i] = metrics.peak_value(inst_acc)
        v_smooth[i] = metrics.smoothness(inst_acc)

        if(len(trials[i]) > max_trial_length):
            max_trial_length = len(trials[i])

    #plt.plot(trials[0][:,0])
    #plt.show()
    n_traj = np.ceil(metrics.avg_traj(trials, max_trial_length))/2

    #import pdb; pdb.set_trace()
    return np.median(v_peak), np.median(vt_peak), np.median(a_peak) , np.median(v_smooth), n_traj
