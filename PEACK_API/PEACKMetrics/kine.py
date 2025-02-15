import numpy as np
import matplotlib.pyplot as plt
import PEACK_Filters as PF
from PEACKMetrics import metrics
from scipy.signal import find_peaks

def velocity(ts, fs, axis = -1):

    v = np.diff(ts,axis = axis)/(1.0/fs)    
    return np.vstack((v[0], v))

def acceleration(ts = None, vs = None, fs = 60, axis = -1): #Can calculate acceleration from position or velocity data

    if not(ts is None) and (vs is None):
        v = velocity(ts, fs, axis)
        a = np.diff(v, axis = axis)/(1.0/fs)
        a = np.vstack((a[0], a))
    elif not(vs is None) and (ts is None):
        a = np.diff(vs, axis = axis)/(1.0/fs)
        a = np.vstack((a[0], a))
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
    v = np.apply_along_axis(PF.butter_lowpass_filter, axis, v, *[0.5, fs, 5])
    #v = np.apply_along_axis(PF.exp_smoothing_vector, axis, v, *[fs, 0.9])
    #speed = np.linalg.norm(v,2, axis=1)

    a = acceleration(None, v, fs, axis=axis)
    a = np.apply_along_axis(PF.median_filter_vector, axis, a, *[fs, 0.1])
    a = np.apply_along_axis(PF.butter_lowpass_filter, axis, a, *[0.5, fs, 5])
    #a = np.apply_along_axis(PF.exp_smoothing_vector, axis, a, *[fs, 0.9])

    return v,a

def get_peaks(ts, prominence):

    idx,_ = find_peaks(ts, distance=None, prominence= prominence/3, plateau_size=[1,15])
    start_idx = idx[:-1]
    end_idx = idx[1:]

    return start_idx, end_idx

def Eval_trajectories(trials, v_trials, a_trials, Fs=60):

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
        a_peak[i] = np.median(np.abs(inst_acc))#metrics.peak_value(inst_acc)
        v_smooth[i] = metrics.smoothness(inst_acc, Fs)

        if(len(trials[i]) > max_trial_length):
            max_trial_length = len(trials[i])

    #plt.plot(trials[0][:,0])
    #plt.show()
    #n_traj = np.ceil(metrics.avg_traj(trials, max_trial_length))/2

    #import pdb; pdb.set_trace()
    #return np.mean(v_peak), np.mean(vt_peak), np.mean(a_peak) , np.mean(v_smooth)
    return v_peak, vt_peak, a_peak , v_smooth
    #return np.median(v_peak), np.median(vt_peak), np.median(a_peak) , np.median(v_smooth), n_traj
    

def load_timestamps(fname):
    
    seg_ts = []
    with open(fname, 'r') as file_ts:
        for line in file_ts:
            str = line.strip()
            str = str.replace(':', '.')
            mm,ss = str.split('.')

            num = int(mm)*60 + int(ss)

            seg_ts.append(num)

    return np.array(seg_ts)