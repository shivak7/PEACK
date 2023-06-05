import numpy as np
from PEACKMetrics.metrics import avg_traj
from matplotlib import pyplot as plt
from PEACKMetrics.segment import trials
from scipy.signal import find_peaks

def wrist_velocity(Body):

    dT = np.median(np.diff(Body.time))

    rw_range = np.std(Body['RWrist'],axis=0)[0]     #std.dev of X-axis for Right wrist
    lw_range = np.std(Body['LWrist'],axis=0)[0]     #std.dev of X-axis for Left wrist

    active_wrist = 'RWrist'
    if(lw_range > rw_range):                #Maybe add  a tolerance limit so bimanual movement is detected? That ideally shouldn't happen for Box and Blocks though!
        active_wrist = 'LWrist'    

    Wrist_vel = np.linalg.norm(np.diff(Body[active_wrist],axis=0),axis=-1)/dT
    Wrist_vel = np.append(0, Wrist_vel)

    
    ts = Body[active_wrist][:,0]
    ts = ts - np.mean(ts)
    prominence = np.std(ts) #+ np.mean(ts)

    idx,_ = find_peaks(ts, distance=None, prominence= prominence/3, plateau_size=[1,15])
    
    start_idx = idx[:-1]
    end_idx = idx[1:]

    plt.plot(Body.time, ts)
    plt.plot(Body.time[start_idx], ts[start_idx], 'r*')
    plt.plot(Body.time[end_idx], ts[end_idx], 'k*')
    plt.show()

    Trials = trials(ts, start_idx, end_idx)
    
    labels = avg_traj(Trials, min_monotonic_range = prominence/3)

    import pdb; pdb.set_trace()