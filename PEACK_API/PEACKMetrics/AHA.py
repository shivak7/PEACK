import numpy as np
from PEACKMetrics.metrics import total_distance_moved, total_angular_distance, angle_between_ts_vectors, angle_between_ts_vector_ref, cumulative_angular_distance
import matplotlib.pyplot as plt
import PEACK_Filters as PF
from PEACKMetrics.kine import Eval_trajectories, velocity, acceleration, va_process
from PEACKMetrics.graphs import plot_double_y
filecount = 0

def reach_duration(ElbAngle, time, reach_angle_thresh = 90, min_gap_threshold = 1, duration_threshold = 2):

    thresh_signal = (ElbAngle > reach_angle_thresh) + 0
    d_thresh_signal = np.hstack((0,np.diff(thresh_signal)))
    reach_starts = time[d_thresh_signal==1]
    reach_ends = time[d_thresh_signal==-1]

    if len(reach_ends)==0 or len(reach_starts)==0:
        #import pdb; pdb.set_trace()
        return np.nan, np.nan

    while reach_ends[0] < reach_starts[0]:     #False ending detected before a start
        reach_ends = np.delete(reach_ends, [0])
        
    n_reaches = np.min([len(reach_starts),len(reach_ends)])
    reach_starts = reach_starts[:n_reaches]     # Set start and end times vectors to the same number of samples 
    reach_ends = reach_ends[:n_reaches]
    
    org_reach_starts = reach_starts.copy()
    org_reach_ends = reach_ends.copy()
    #   Merge durations that are close to each other
    min_gap_threshold = 1   #1 second
    no_small_gaps_flag = False
    
    while no_small_gaps_flag == False:  #Repeat merging until there are no small gaps between events left
    
        no_small_gaps_flag = True
        start_gaps = np.diff(reach_starts)
        new_reach_starts = []
        new_reach_starts.append(reach_starts[0])
        new_reach_ends = []
        new_reach_ends.append(reach_ends[0])
        
        for i in range(1,len(reach_starts)):

            if start_gaps[i-1] > min_gap_threshold:         #All good, store start and stop in new list
                new_reach_starts.append(reach_starts[i])
                new_reach_ends.append(reach_ends[i])
            else:                                           # Merge event with previous event
                #import pdb; pdb.set_trace()
                new_reach_ends[-1] = reach_ends[i]
                no_small_gaps_flag = False

        reach_starts = new_reach_starts.copy()
        reach_ends = new_reach_ends.copy()

    #   Remove reach durations that are too short (attributed to noisy tracking)
    #   default duration_threshold = 2 seconds

    reach_starts = np.array(reach_starts)
    reach_ends = np.array(reach_ends)
    del_idx = []

    for dur_idx in range(len(reach_starts)):

        duration_in_sec = reach_ends[dur_idx] - reach_starts[dur_idx]
        if duration_in_sec < 0:
            print("Something has gone seriously wrong in detecting start and end times of reach durations!")
            raise SystemError
        
        if duration_in_sec < duration_threshold:
            del_idx.append(dur_idx)


    reach_starts = np.delete(reach_starts, del_idx)
    reach_ends = np.delete(reach_ends, del_idx)

    return reach_starts, reach_ends
    # plt.figure()
    # plt.plot(time, ElbAngle)
    # #plt.plot(time, thresh_signal*reach_angle_thresh)
    # for idx in range(len(reach_starts)):
    #     plt.plot([reach_starts[idx], reach_starts[idx]],[0, 120],'r')
    #     plt.plot([reach_ends[idx], reach_ends[idx]],[0, 120],'k')
    # plt.show()
def filter_by_angular_velocity(angle_ts, dT, filter_len=0.5):

    Fs = int(np.round(1/dT))
    AngVel_Elb = np.diff(angle_ts)/dT
    fAV = PF.median_filter_vector(AngVel_Elb, Fs, mult=filter_len) #Median filter out impossibly large angular velocities (window size is critical for this)
    #fAV = PF.butter_lowpass_filter(AngVel_Elb, 1, Fs, 5)
    ifAV = np.hstack((angle_ts[0], fAV))
    ifAV = np.cumsum(ifAV*dT)                                      # Integrate ang. velocity ts to get back ang. position
    difAV = PF.sdetrend(ifAV) + angle_ts[0]
    return difAV


def reach_angular_metrics(ElbAngle, t0, t_start, t_end, Fs):

    ret_vals = []
    for i in range(len(t_start)):
        start_idx = int((t_start[i] - t0)*Fs)
        end_idx = int((t_end[i] - t0)*Fs)
        signal = ElbAngle[start_idx:end_idx]
        peak_thresh = np.max(signal)*0.95
        sig_dist = np.abs((peak_thresh - signal))
        thresh_idx = np.where(sig_dist<=2)

        first_thresh = thresh_idx[0][0] 
        last_thresh = thresh_idx[0][-1]
        time_to_ext = first_thresh / Fs
        time_to_flex = (len(signal) - last_thresh)/Fs
        ext_ang_vel = (signal[first_thresh] - signal[0])/time_to_ext
        flex_ang_vel = (signal[last_thresh] - signal[-1])/time_to_flex
        grasp_time = (last_thresh - first_thresh)/Fs
        X = [time_to_ext, time_to_flex, ext_ang_vel, flex_ang_vel, grasp_time]
        ret_vals.append(X)
    
    return ret_vals


def reach_metrics(Body, Win=[]):

    try:
        RShoulderElbow = Body["RShoulder"] - Body["RElbow"]
        LShoulderElbow = Body["LShoulder"] - Body["LElbow"]

        RWristElbow = Body["RWrist"] - Body["RElbow"]
        LWristElbow = Body["LWrist"] - Body["LElbow"]
    

        dT = np.median(np.diff(Body.time))
        Fs = int(np.round(1/dT))
        
        degScaleFactor = 180/np.pi
        RElbAngle = degScaleFactor*angle_between_ts_vectors(RShoulderElbow, RWristElbow)
        LElbAngle = degScaleFactor*angle_between_ts_vectors(LShoulderElbow, LWristElbow)
        
        #plt.plot(Body.time/1000, RElbAngle); plt.show()

        print(Body.filename)
        r_starts, r_ends = reach_duration(RElbAngle, Body.time, reach_angle_thresh = 100, min_gap_threshold = 0.1, duration_threshold = 1)
        l_starts, l_ends = reach_duration(LElbAngle, Body.time, reach_angle_thresh = 100, min_gap_threshold = 0.1, duration_threshold = 1)
        
        r_ret_vals = reach_angular_metrics(RElbAngle, Body.time[0], r_starts, r_ends, Fs)
        l_ret_vals = reach_angular_metrics(LElbAngle, Body.time[0], l_starts, l_ends, Fs)

        r_reach_metrics = np.nanmedian(r_ret_vals, axis=0)
        l_reach_metrics = np.nanmedian(l_ret_vals, axis=0)

        if np.isnan(r_reach_metrics).all():
            r_reach_metrics = [np.nan]*5
        if np.isnan(l_reach_metrics).all():
            l_reach_metrics = [np.nan]*5
    
    except:
        return np.array([np.nan]*10)
    # plt.figure()
    # plt.plot(RElbAngle)
    # plt.plot(LElbAngle)
    # plt.show()
    #import pdb; pdb.set_trace()
    return np.hstack((l_reach_metrics, r_reach_metrics))
    

    #import pdb; pdb.set_trace()

    #skip_len = 0
    #plot_double_y(RElbAngle[skip_len:], rw_speed[skip_len:], Body.time[skip_len:])
    #plot_double_y(LElbAngle[skip_len:], lw_speed[skip_len:], Body.time[skip_len:])
    #plt.show()
    #import pdb; pdb.set_trace()
    
    
def reach_stats_v2(Body, Win=[]):

    try:
        dT = np.median(np.diff(Body.time))
        rw_v_ts = velocity(Body["RWrist"], 1/dT, axis=0)
        lw_v_ts = velocity(Body["LWrist"], 1/dT, axis=0)
        rw_speed = np.linalg.norm(rw_v_ts,axis=1)
        lw_speed = np.linalg.norm(lw_v_ts,axis=1)
        lw_reach = np.trapz(lw_speed)/len(lw_speed)
        rw_reach = np.trapz(rw_speed)/len(rw_speed)
    except:
        return np.nan, np.nan

    return lw_reach, rw_reach


def trunk_displacement_angle(Body, mode='statistic', side=''):

    try:
        TrunkVec = Body["Neck"] - Body["MidHip"]
        #ShouldersVec = Body["RShoulder"] - Body["LShoulder"]
        ShouldersVec = Body["LShoulder"] - Body["Neck"]
        #import pdb; pdb.set_trace()
        #Angle = total_angular_distance(Body["Neck"], Body["MidHip"])  #angle_between_ts_vectors(TrunkVec, ShouldersVec)
        #Angle = cumulative_angular_distance(Body["Neck"], Body["MidHip"])
        # R = np.sqrt(np.sum(np.square(TrunkVec),axis=1))
        # S = np.diff(Body["Neck"], axis=0)
        # S = np.linalg.norm(S,axis=1)
        # S = np.hstack((0, S))
        # Angle2 = S/R
        #Angle3 = angle_between_ts_vectors(TrunkVec, ShouldersVec)*180/np.pi
        Angle = angle_between_ts_vector_ref(TrunkVec,np.median(TrunkVec, axis=0))
        # dAngle3 = np.diff(Angle3, axis=0)
        # dAngle3 = np.hstack([0, dAngle3])
        # #ShoulderLen = np.sqrt(np.sum(np.square(ShouldersVec),axis=1))
        # HipMvmt = np.diff(Body["MidHip"], axis=0)
        # HipMvmt = np.linalg.norm(HipMvmt,axis=1)
        # HipMvmt = np.hstack([0, HipMvmt])
        #import pdb; pdb.set_trace()
        # #plt.plot(HipMvmt)
        # #plt.figure()
        # plt.plot(Angle)
        # plt.plot(Angle2)
        # plt.plot(np.abs(dAngle3))
        # #plt.plot(TrunkLen/200)
        # plt.show()
        #return Angle3 #- np.abs(dAngle3)
        #import pdb; pdb.set_trace()
        #return TrunkLen/ShoulderLen
        #return np.mean(Angle)/np.std(Angle)
        if mode=='raw':
            return Angle
        elif mode=='statistic':
            return np.sum(Angle)/len(Angle)
        else:
            print('Invalid return mode selected for kinematic metric. Please choose mode=\'statistic\' or \'raw\'')
        #Angle = angle_between_ts_vectors(TrunkVec, TrunkVec[0].reshape(1,-1))
        #Angle = Angle - np.mean(Angle)
        #Angle = np.square(Angle)
        Angle2 = Angle
        #Angle = Angle[int(Win[0]*Body.fs) : int(Win[1]*Body.fs)]
        Angle = Angle[~np.isnan(Angle)]
        Angle = np.sort(Angle)
        #import pdb; pdb.set_trace()
        
        
        l1 = int(len(Angle)*0.1)       # Take top 10 percentile of values
        ul = np.nanmedian(Angle[-l1:])
        bl = np.nanmedian(Angle[:l1])
        angle_range = ul - bl
        # print(angle_range)
        # plt.plot(Angle2)
        # plt.show()
        #import pdb; pdb.set_trace()
        #Angle3 = PF.butter_highpass_filter(Angle2.reshape(-1,1), 1, Body.fs, 3)
        #Angle3 = PF.position_median_filter(np.reshape(Angle2,-1,1), Body.fs, 1)
        global filecount
        if Body.type == 'VICON':
            filehead = 'VFile'
        else:
            filehead = 'PFile'

        # print(Body.filename)
        # fname = filehead + str(filecount) + '.txt'
        # np.savetxt(fname, [Body.time, Angle2])
        # filecount += 1
        #print('Movement range : ', ul-bl)
        #plt.plot(Angle)
        #plt.show()
        
        return np.std(Angle)#angle_range
        #return np.nanmean(Angle)
        #return np.nanstd(Angle)    # Original default value
    except:
        return np.nan

def trunk_rotation_angle(Body, mode='statistic', side=''):

    try:
        #ThetaSh = total_angular_distance(Body["RShoulder"], Body["LShoulder"])      #Trunk rotation: sum absolute change vs std
        ShouldersVec = Body["LShoulder"] - Body["RShoulder"]
        ThetaSh = angle_between_ts_vector_ref(ShouldersVec,np.median(ShouldersVec, axis=0))
        #import pdb; pdb.set_trace()
        #plt.plot(ThetaSh)
        #plt.show()
        if mode=='raw':
            return ThetaSh
        elif mode=='statistic':
            return np.sum(ThetaSh)/len(ThetaSh)
        else:
            print('Invalid return mode selected for kinematic metric. Please choose mode=\'statistic\' or \'raw\'')
        
        
    except:
        return np.nan

def trunk_displacement_distance(Body, Win=[]):

    

    try:
        RSh = total_distance_moved(Body["RShoulder"])
        LSh = total_distance_moved(Body["LShoulder"])
        Ref = Body["RShoulder"] - Body["RElbow"]
    except:

        try:
            RSh = total_distance_moved(Body["RShoulder"])
            LSh = total_distance_moved(Body["LShoulder"])
            Ref = Body["LShoulder"] - Body["LElbow"]
        except:
            return np.nan

    RefLength = np.nanmedian(np.linalg.norm(Ref, axis=1))
    
    #Res1 = ThetaSh #RSh + LSh
    #import pdb; pdb.set_trace()
    # RElb = total_distance_moved(Body["RElbow"])
    # try:
    #     LElb = total_distance_moved(Body["LElbow"])
    # except:
    #     import pdb; pdb.set_trace()

    # Res2 = np.max([RElb,LElb])

    # Res = Res2#[Res1, Res2]
    #return Res
    #print(RefLength)
    return (RSh + LSh)/(RefLength)


def elbow_flexion_angle(Body, mode='statistic'):

    try:

        RShoulderElbow = Body["RShoulder"] - Body["RElbow"]
        LShoulderElbow = Body["LShoulder"] - Body["LElbow"]

        RWristElbow = Body["RWrist"] - Body["RElbow"]
        LWristElbow = Body["LWrist"] - Body["LElbow"]

        RElbAngle = angle_between_ts_vectors(RShoulderElbow, RWristElbow)
        LElbAngle = angle_between_ts_vectors(LShoulderElbow, LWristElbow)

        #import pdb; pdb.set_trace()
        RElbAngle = PF.position_median_filter(RElbAngle.reshape(-1,1), Body.fs, 0.5)
        LElbAngle = PF.position_median_filter(LElbAngle.reshape(-1,1), Body.fs, 0.5)
        plt.plot(LElbAngle)
        plt.plot(RElbAngle)
        plt.show()
        import pdb; pdb.set_trace()
        
        # ratio = np.nanmedian(RElbAngle / LElbAngle)
        # angle_ratio = ratio if ratio < 1 else 1/ratio
        # return angle_ratio, angle_ratio

        RElbAngle = np.sort(RElbAngle)
        RElbAngle[np.isnan(RElbAngle)] = []
        LElbAngle = np.sort(LElbAngle)
        LElbAngle[np.isnan(LElbAngle)] = []
        
        l1 = int(len(RElbAngle)*0.05)       # Take top 5 percentile of values
        l2 = int(len(LElbAngle)*0.05)

        
        if(l1 <= 5):
            RElbAngle95 = np.median(RElbAngle)
        else:
            RElbAngle95 = np.median(RElbAngle[-l1:])
        
        if(l2 <= 5):
            LElbAngle95 = np.median(LElbAngle)
        else:
            LElbAngle95 = np.median(LElbAngle[-l1:])

        return LElbAngle95, RElbAngle95 
        #return np.median(LElbAngle), np.median(RElbAngle)
    
    except:
        #return np.nan
        return [np.nan, np.nan]


def hand_coordination(Body, args=[]):

    
    try:
        dT = np.median(np.diff(Body.time))
        Fs = int(1/dT)

        rwrist = Body['RWrist']
        lwrist = Body['LWrist']

        try:
            rw_v_ts = velocity(rwrist, 1/dT, axis=0)
        except:
            return 0
        
        try:
            lw_v_ts = velocity(lwrist, 1/dT, axis=0)
        except:
            return 0
        
        rw_v_inst = np.linalg.norm(rw_v_ts, axis=1)
        lw_v_inst = np.linalg.norm(lw_v_ts, axis=1)

        rw_v_inst = rw_v_inst[:-Fs]
        lw_v_inst = lw_v_inst[:-Fs]
        #rw_v_inst = (rw_v_inst - np.mean(rw_v_inst)) / np.std(rw_v_inst)
        #lw_v_inst = (lw_v_inst - np.mean(lw_v_inst)) / np.std(lw_v_inst)
        
        rw_v_inst = rw_v_inst / np.sum(rw_v_inst)
        lw_v_inst = lw_v_inst / np.sum(lw_v_inst)

        coordination_ratio = (np.trapz(rw_v_inst * lw_v_inst)) / (np.trapz(rw_v_inst) + np.trapz(lw_v_inst))
        #import pdb; pdb.set_trace()
        if coordination_ratio > 1:
            print("Bad coordination ratio detected!")
            return np.nan
        return coordination_ratio
        # print(Body.filename)
        # plt.plot(rw_v_inst)
        # plt.plot(lw_v_inst)
        # plt.show()
        #import pdb; pdb.set_trace()
    except:

        return np.nan