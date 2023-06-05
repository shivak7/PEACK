import numpy as np
from PEACKMetrics.metrics import avg_traj
from matplotlib import pyplot as plt
from PEACKMetrics.segment import trials
from scipy.signal import find_peaks
from PEACKMetrics.kine import Eval_trajectories, velocity, acceleration

def wrist_velocity(Body):

    dT = np.median(np.diff(Body.time))
    Fs = int(1/dT)

    rwrist = Body['RWrist']
    lwrist = Body['LWrist']

    if(len(lwrist)==0):
        active_wrist = 'RWrist'
    elif (len(rwrist)==0):
        active_wrist = 'LWrist'
    else:
        rw_range = np.std(rwrist[Fs:-Fs], axis=0)[0]            #std.dev of X-axis for Right wrist
        lw_range = np.std(lwrist[Fs:-Fs],axis=0)[0]             #std.dev of X-axis for Left wrist

        active_wrist = 'RWrist'
        if(lw_range > rw_range):                #Maybe add  a tolerance limit so bimanual movement is detected? That ideally shouldn't happen for Box and Blocks though!
            active_wrist = 'LWrist'    

    # Wrist_vel = np.linalg.norm(np.diff(Body[active_wrist],axis=0),axis=-1)/dT
    # Wrist_vel = np.append(0, Wrist_vel)

    ts = Body[active_wrist][:,0]
    ts = ts - np.mean(ts)
    prominence = np.std(ts) #+ np.mean(ts)
    idx,_ = find_peaks(ts, distance=None, prominence= prominence/3, plateau_size=[1,15])
    start_idx = idx[:-1]
    end_idx = idx[1:]

    # plt.plot(Body.time, ts)
    # plt.plot(Body.time[start_idx], ts[start_idx], 'r*')
    # plt.plot(Body.time[end_idx], ts[end_idx], 'k*')
    # plt.show()

    ts = Body[active_wrist]

    v_ts = velocity(ts, 1/dT, axis=0)
    a_ts = acceleration(None, v_ts, 1/dT, axis=0)

    Trials = trials(ts, start_idx, end_idx)
    v_trials = trials(v_ts, start_idx, end_idx)
    a_trials = trials(a_ts, start_idx, end_idx)
    
    peak_vel, time_to_peak_vel, acc, smooth = Eval_trajectories(Trials, v_trials, a_trials)
    #for i in range(len(Trials)):
    #   plt.plot(Trials[i])
    #plt.show()
    
    #labels = avg_traj(Trials, min_monotonic_range = prominence/6)
    #import pdb; pdb.set_trace()

    return peak_vel/1000, time_to_peak_vel, acc/1000, smooth/1000