import numpy as np
from PEACKMetrics.metrics import total_distance_moved, total_angular_distance, angle_between_ts_vectors, cumulative_angular_distance
import matplotlib.pyplot as plt
from PEACKMetrics.segment import trials
from PEACKMetrics import kine #import Eval_trajectories, velocity, acceleration, va_process
from scipy.signal import hilbert
from scipy.stats import zscore

def wrist_velocity(Body):

    dT = np.median(np.diff(Body.time))
    Fs = int(1/dT)

    rwrist = Body['RWrist']
    lwrist = Body['LWrist']

    
    rw_v_ts = kine.velocity(rwrist, 1/dT, axis=0)
    lw_v_ts = kine.velocity(lwrist, 1/dT, axis=0)
    
    print(Body.filename)
    # plt.subplot(1,2,1)
    # plt.plot(rw_v_ts)
    # plt.subplot(1,2,2)
    # plt.plot(lw_v_ts)
    # plt.show()
    rw_med = np.median(rw_v_ts)
    lw_med = np.median(lw_v_ts)

    #labels = avg_traj(Trials, min_monotonic_range = prominence/6)
    #import pdb; pdb.set_trace()

    return lw_med, rw_med #, time_to_peak_vel, acc/1000, smooth



def trunk_displacement_angle(Body, Win=[15,-15]):

    try:
        TrunkVec = Body["Neck"] - Body["MidHip"]
        #ShouldersVec = Body["RShoulder"] - Body["LShoulder"]
        ShouldersVec = Body["LShoulder"] - Body["Neck"]
        #Angle = total_angular_distance(Body["Neck"], Body["MidHip"])  #angle_between_ts_vectors(TrunkVec, ShouldersVec)
        Angle = cumulative_angular_distance(Body["Neck"], Body["MidHip"])
        return np.sum(Angle)/len(Angle)
    except:
        return np.nan

def hand_coordination(Body, args=[]):

    dT = np.median(np.diff(Body.time))
    Fs = int(1/dT)

    rwrist = Body['RWrist']
    lwrist = Body['LWrist']

    
    rw_v_ts = kine.velocity(rwrist, 1/dT, axis=0)
    lw_v_ts = kine.velocity(lwrist, 1/dT, axis=0)
    rw_v_inst = np.linalg.norm(rw_v_ts, axis=1)
    lw_v_inst = np.linalg.norm(lw_v_ts, axis=1)

    rw_v_inst = (rw_v_inst - np.mean(rw_v_inst)) / np.std(rw_v_inst)
    lw_v_inst = (lw_v_inst - np.mean(lw_v_inst)) / np.std(lw_v_inst)
    
    coordination_ratio = np.trapz(rw_v_inst * lw_v_inst) / (np.trapz(rw_v_inst) + np.trapz(lw_v_inst))

    # print(Body.filename)
    # plt.plot(rw_v_inst)
    # plt.plot(lw_v_inst)
    # plt.show()
    #import pdb; pdb.set_trace()
    
    return coordination_ratio

def wrist_segmented_metrics(Body, mode='statistic', side=''):

    dT = np.median(np.diff(Body.time))
    #import pdb; pdb.set_trace()
    Fs = int(1/dT)

    if side=='':

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
    elif side == 'L' or side == 'R':
        active_wrist = side + 'Wrist'
                
    #wrist_ts = Body[active_wrist]
    rwrist_ts = Body['RWrist']
    lwrist_ts = Body['LWrist']
    #rw_v_ts = kine.velocity(rwrist_ts, 1/dT, axis=0)
    #lw_v_ts = kine.velocity(lwrist_ts, 1/dT, axis=0)

    rw_v_ts = kine.acceleration(ts=rwrist_ts, fs=1/dT, axis=0)
    lw_v_ts = kine.acceleration(ts=lwrist_ts, fs=1/dT, axis=0)

    rw_v_inst = np.linalg.norm(rw_v_ts, axis=1)
    lw_v_inst = np.linalg.norm(lw_v_ts, axis=1)
    
    #ts1 = rw_v_inst #rw_v_inst
    #h_ts1 = hilbert(ts1)
    #ts1_env = np.abs(h_ts1)

    #ts2 = lw_v_inst #rw_v_inst
    #h_ts2 = hilbert(ts2)
    #ts2_env = np.abs(h_ts2)
    
    ts1 = Body['RWrist'][:,0]
    ts2 = Body['RWrist'][:,1]
    #ts = ts - np.mean(ts)

    ts_segs = kine.load_timestamps('DataFiles/BT36_segments.txt')
    #prominence = np.std(ts) #+ np.mean(ts)
    #start_idx, end_idx = kine.get_peaks(ts, prominence)

    plt.plot(Body.time, ts1)
    #plt.plot(Body.time, ts1_env,'r')
    
    #plt.figure()
    plt.plot(Body.time, ts2)
    #plt.plot(Body.time, ts2_env,'r')

    y1 = 0*np.ones_like(ts1)#np.max(ts1)*np.ones_like(ts1)
    y2 = 1200*np.ones_like(ts1)#np.min(ts1)*np.ones_like(ts1)

    for temp_i in range(len(ts_segs)):

        c_t = ts_segs[temp_i]
        y1_t = y1[temp_i]
        y2_t = y2[temp_i]
        
        plt.plot([c_t, c_t], [y1_t, y2_t],'k--')


    #plt.figure()
    #plt.plot(Body.time, zscore(ts1))
    #plt.plot(Body.time, zscore(ts2))
    #plt.plot(Body.time[start_idx], ts[start_idx], 'r*')
    #plt.plot(Body.time[end_idx], ts[end_idx], 'k*')
    plt.show()

    import pdb; pdb.set_trace()
    ts = Body[active_wrist]

    # v_ts = velocity(ts, 1/dT, axis=0)
    # a_ts = acceleration(None, v_ts, 1/dT, axis=0)

    # Trials = trials(ts, start_idx, end_idx)
    # v_trials = trials(v_ts, start_idx, end_idx)
    # a_trials = trials(a_ts, start_idx, end_idx)
    
    peak_vel, time_to_peak_vel, acc, smooth = kine.Eval_trajectories(Trials, v_trials, a_trials)
    #for i in range(len(Trials)):
    #   plt.plot(Trials[i])
    #plt.show()
    
    #labels = avg_traj(Trials, min_monotonic_range = prominence/6)
    #import pdb; pdb.set_trace()

    stat = np.median
    return stat(peak_vel/1000), stat(time_to_peak_vel), stat(acc/1000), stat(smooth)

#list of initial variables (1 & 2 have multiple joints per variable):

    # 1) degrees of freedom: how many joint angles across the shoulders, elbows, wrists, and digits are moving during task execution?
    #(looking at Bernstein's freezing DoF)
    
    #DoF including Joint rotations may be a bit challenging
        #Do you think we could look at total number of moving joints (in the UE) without looking at specific joint angles? e.g., how many joints
        #across the wrist, elbow, and shoulder (and potentially trunk) are engaged during activity? If not, we can omit this variable.
    
    # 2) mean velocity: amplitude of mean velocities of wrist and shoulder movements during the full task trial

    # Works like the wrist_velocity function above 
    
    # 3) goal synchronization: time difference between first wrist moving towards a cup stack and second wrist moving
    #towards the cup stack (6 repetitions of this across each trial, can we distinguish between assembly and disassembly?)

    #We can Try! I think this is quite doable. We can basically look at time lag to movement onset. Would that be OK?
        #Yes, great! I think that would capture what I would like to measure.

    # 4) movement overlap: time that both hands are moving simtultaneously and engaged in the task as a percentage of total
    #task completion time (intersection of the area under the curve)

    # I will have to look into writing this out perhaps.

    # 5) trunk displacement: trunk displacement angle and distance during task execution 
    
    # I think we already have this in AHA

