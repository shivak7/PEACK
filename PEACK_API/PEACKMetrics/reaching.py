import numpy as np
from PEACKMetrics.metrics import angle_between_ts_vectors, total_distance_moved
from matplotlib import pyplot as plt
from PEACKMetrics.segment import trials
from scipy.signal import find_peaks
from PEACKMetrics.kine import Eval_trajectories, velocity, acceleration, va_process
#from Render.skeleton import plot_body, animate_body


def find_active_hand(Body, win=[0,-1]):

    dT = np.median(np.diff(Body.time))
    Fs = int(1/dT)

    rwrist = Body['RWrist']#[win[0]:win[1],:]
    lwrist = Body['LWrist']#[win[0]:win[1],:]
    if(len(lwrist)==0):
        active_side = 'R'
    elif (len(rwrist)==0):
        active_side = 'L'
    else:
        #rw_range = np.std(rwrist[Fs:-Fs], axis=0)[0]            #std.dev of X-axis for Right wrist
        #lw_range = np.std(lwrist[Fs:-Fs], axis=0)[0]             #std.dev of X-axis for Left wrist
        
        rw_range = total_distance_moved(rwrist[win[0]:win[1],:])
        lw_range = total_distance_moved(lwrist[win[0]:win[1],:])
        active_side = 'R'
        if(lw_range > rw_range):                #Maybe add  a tolerance limit so bimanual movement is detected? That ideally shouldn't happen for Box and Blocks though!
            active_side = 'L'    

    return active_side

def metrics_raw(Body):

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
        lw_range = np.std(lwrist[Fs:-Fs], axis=0)[0]             #std.dev of X-axis for Left wrist

        active_wrist = 'RWrist'
        if(lw_range > rw_range):                #Maybe add  a tolerance limit so bimanual movement is detected? That ideally shouldn't happen for Box and Blocks though!
            active_wrist = 'LWrist'    
    dims = Body[active_wrist].shape[1]
    ts = Body[active_wrist]
    #v_ts = velocity(ts, 1/dT, axis=0)
    #a_ts = acceleration(None, v_ts, 1/dT, axis=0)
    v_ts, a_ts = va_process(ts, Fs, axis = 0)

    #import pdb; pdb.set_trace()
    inst_vel = np.linalg.norm(v_ts,axis=-1)
    return inst_vel#, v_ts, a_ts


def elbow_angle(Body, mode='statistic', side=None):

    if side==None:
        active_side = find_active_hand(Body)            # Returns L or R depending on side that has the most movement activity. Useful only for Unimanual tasks
    else:
        active_side = side

    #animate_body(Body)
    
    ActElbow = active_side + "Elbow"
    ActShoulder = active_side + "Shoulder"
    ActWrist = active_side + "Wrist"
    
    try:
        ShoulderElbow = Body[ActShoulder] - Body[ActElbow]
        WristElbow = Body[ActWrist] - Body[ActElbow]
    except:
        return []
    
    ElbAngle = angle_between_ts_vectors(ShoulderElbow,WristElbow)

    if mode=='statistic':
        return np.median(ElbAngle)
    elif mode=='raw':
        return ElbAngle
    else:
        print('Invalid return mode selected for kinematic metric. Please choose mode=\'statistic\' or \'raw\'')
    #return np.median(ElbAngle)


def metrics(Body):

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
    dims = Body[active_wrist].shape[1]
    ts = Body[active_wrist]
    #v_ts = velocity(ts, 1/dT, axis=0)
    #a_ts = acceleration(None, v_ts, 1/dT, axis=0)
    v_ts, a_ts = va_process(ts, Fs, axis = 0)

    ts = ts.reshape(1, -1, dims)
    v_ts = v_ts.reshape(1, -1, dims)
    a_ts = a_ts.reshape(1, -1, dims)

    peak_vel, time_to_peak_vel, acc, smooth = Eval_trajectories(ts, v_ts, a_ts, Fs)

    #import pdb; pdb.set_trace()
    #plt.plot(rwrist); plt.show()
    return peak_vel, time_to_peak_vel, acc, smooth
    #inst_vel = np.linalg.norm(v_ts,axis=-1)
    #return inst_vel#, v_ts, a_ts
