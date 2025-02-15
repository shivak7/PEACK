from statistics import mean
from weakref import ref
import numpy as np
import scipy
import pickle
import variableDeclaration as var
from scipy.signal import find_peaks
import os
import matplotlib.pyplot as plt
from reaching_validation_gui import crosscorr
import warnings
warnings.filterwarnings("ignore")
os.system('cls' if os.name == 'nt' else 'clear') # clears terminal

'''
    Copyright 2022 Jan Breuer & Shivakeshavan Ratnadurai-Giridharan

    A set of functions to calculate different movement assessment parameters
'''

# calculate the time to peak values
def time_to_peak(all_data_dict, participant, trial, mov_seg,joint,joint_selection,condition="VICON", debug = False):
    # create time vector specifically for this segment
    if condition == "VICON":
        seg_start = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][1]
        seg_time = all_data_dict["VICON_time"][participant][trial][seg_start:seg_end]
        # calculate the index of the peak
        arg_peak = seg_time[np.argmax(all_data_dict["VICON_velocity_segmented"][participant][trial][joint][mov_seg])]
        peak = np.max(all_data_dict["VICON_velocity_segmented"][participant][trial][joint][mov_seg])
        mean_vel = np.mean(all_data_dict["VICON_velocity_segmented"][participant][trial][joint][mov_seg])
        ttp = arg_peak-seg_time[0] #time of peak - start time

    elif condition == "PEACK":
        seg_start = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][1]
        seg_time = all_data_dict["PEACK_time"][participant][trial][seg_start:seg_end]
        # calculate the index of the peak
        arg_peak = seg_time[np.argmax(all_data_dict["PEACK_velocity_segmented"][participant][trial][joint][mov_seg])]
        peak = np.max(all_data_dict["PEACK_velocity_segmented"][participant][trial][joint][mov_seg])
        mean_vel = np.mean(all_data_dict["PEACK_velocity_segmented"][participant][trial][joint][mov_seg])
        ttp = arg_peak-seg_time[0] #time of peak - start time

    if debug:
        print(condition)
        print("Time to peak", ttp)
        print("Peak",peak)
        print("Mean Vel", mean_vel)
    return [round(ttp,2), round(peak,3), round(mean_vel,3)]

# calculate the total path values
def total_path_calc(all_data_dict, participant, trial, mov_seg,joint,joint_selection,condition="VICON"):
    if condition == "VICON":
        seg_start = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][1]
        euc_dist = np.linalg.norm(np.diff(all_data_dict["VICON_positional"][participant][trial][joint][seg_start:seg_end], axis = 0),axis=1)
        total_path = np.sum(euc_dist)
    elif condition == "PEACK":
        seg_start = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][1]
        euc_dist = np.linalg.norm(np.diff(all_data_dict["PEACK_positional"][participant][trial][joint][seg_start:seg_end], axis = 0),axis=1)
        total_path = np.sum(euc_dist)
    return total_path

def calc_mu(all_data_dict, participant, trial, mov_seg,joint,joint_selection, condition):
    #to calculate an average position across trials, the trials must have the same length - looking for shortest trial
    trial_len = []
    if condition == "VICON":
        for i in range(0,len(all_data_dict["VICON_positional"][participant])): # looping through all trials and figuring out their lengths
            seg_start = all_data_dict["VICON_stamps"][participant][i][joint_selection][mov_seg][0]
            seg_end = all_data_dict["VICON_stamps"][participant][i][joint_selection][mov_seg][1]
            trial_len.append(len(all_data_dict["VICON_positional"][participant][i][joint][seg_start:seg_end]))
        # create empty mu-lists for later appending    
        mu_x = [ [] for _ in range(0,np.min(trial_len)) ]
        mu_y = [ [] for _ in range(0,np.min(trial_len)) ]
        mu_z = [ [] for _ in range(0,np.min(trial_len)) ]
        for i in range(0,len(all_data_dict["VICON_positional"][participant])):
            # calculate a reference for the trial with the shortest length
            ref_start = all_data_dict["VICON_stamps"][participant][np.argmin(trial_len)][joint_selection][mov_seg][0]
            ref_end = all_data_dict["VICON_stamps"][participant][np.argmin(trial_len)][joint_selection][mov_seg][1]
            ref_time = all_data_dict["VICON_time"][participant][np.argmin(trial_len)][ref_start:ref_end]
            seg_start = all_data_dict["VICON_stamps"][participant][i][joint_selection][mov_seg][0]
            seg_end = all_data_dict["VICON_stamps"][participant][i][joint_selection][mov_seg][1]
            seg_time = all_data_dict["VICON_time"][participant][i][seg_start:seg_end]
            
            xCoord = []; yCoord = []; zCoord = []; # saving xyz coordinates independently
            for coord in all_data_dict["VICON_positional"][participant][i][joint][seg_start:seg_end]:
                xCoord.append(coord[0]); yCoord.append(coord[1]); zCoord.append(coord[2])
            # bringing them on the length of the smallest trial
            temp_x = np.interp(ref_time,seg_time,xCoord)
            temp_y = np.interp(ref_time,seg_time,yCoord)
            temp_z = np.interp(ref_time,seg_time,zCoord)
            # appending the mu-coordinates per trial for every point in time (j) 
            for j in range(0, len(temp_x)):
                mu_x[j].append(temp_x[j])
                mu_y[j].append(temp_y[j]) 
                mu_z[j].append(temp_z[j]) 

    elif condition == "PEACK":
        for i in range(0,len(all_data_dict["PEACK_positional"][participant])):
            seg_start = all_data_dict["PEACK_stamps"][participant][i][joint_selection][mov_seg][0]
            seg_end = all_data_dict["PEACK_stamps"][participant][i][joint_selection][mov_seg][1]
            trial_len.append(len(all_data_dict["PEACK_positional"][participant][i][joint][seg_start:seg_end]))
        # create empty mu-lists for later appending   
        mu_x = [ [] for _ in range(0,np.min(trial_len)) ]
        mu_y = [ [] for _ in range(0,np.min(trial_len)) ]
        mu_z = [ [] for _ in range(0,np.min(trial_len)) ]

        for i in range(0,len(all_data_dict["PEACK_positional"][participant])):
             # calculate a reference for the trial with the shortest length
            ref_start = all_data_dict["PEACK_stamps"][participant][np.argmin(trial_len)][joint_selection][mov_seg][0]
            ref_end = all_data_dict["PEACK_stamps"][participant][np.argmin(trial_len)][joint_selection][mov_seg][1]
            ref_time = all_data_dict["PEACK_time"][participant][np.argmin(trial_len)][ref_start:ref_end]
            seg_start = all_data_dict["PEACK_stamps"][participant][i][joint_selection][mov_seg][0]
            seg_end = all_data_dict["PEACK_stamps"][participant][i][joint_selection][mov_seg][1]
            seg_time = all_data_dict["PEACK_time"][participant][i][seg_start:seg_end]
            
            xCoord = []; yCoord = []; zCoord = []; # saving xyz coordinates independently
            for coord in all_data_dict["PEACK_positional"][participant][i][joint][seg_start:seg_end]:
                xCoord.append(coord[0]); yCoord.append(coord[1]); zCoord.append(coord[2])
            # bringing them on the length of the smallest trial
            temp_x = np.interp(ref_time,seg_time,xCoord)
            temp_y = np.interp(ref_time,seg_time,yCoord)
            temp_z = np.interp(ref_time,seg_time,zCoord)
            # appending the mu-coordinates per trial for every point in time (j) 
            for j in range(0, len(temp_x)):
                mu_x[j].append(temp_x[j])
                mu_y[j].append(temp_y[j]) 
                mu_z[j].append(temp_z[j]) 
    # calculating the mean mu-values across coordinates
    for j in range(0, len(temp_x)):
        mu_x[j] = np.average(mu_x[j])
        mu_y[j] = np.average(mu_y[j])
        mu_z[j] = np.average(mu_z[j])
    # returns the averga mu coordinates across trials and the reference time of the smallest trial
    return [mu_x, mu_y, mu_z, ref_time]

def calc_trajectory_var(all_data_dict, participant, trial, mov_seg,joint,joint_selection, condition = "VICON"):
    [mu_x,mu_y,mu_z,ref_time]= calc_mu(all_data_dict, participant, trial, mov_seg,joint,joint_selection, condition)
    if condition == "VICON":
        seg_start = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][1]
        seg_time = all_data_dict["VICON_time"][participant][trial][seg_start:seg_end]
        # storin the positional coordinates independly 
        xCoord = []; yCoord = []; zCoord = [];
        for coord in all_data_dict["VICON_positional"][participant][trial][joint][seg_start:seg_end]:
            xCoord.append(coord[0]); yCoord.append(coord[1]); zCoord.append(coord[2])
        # bringing them one the same length as the reference time of the shortest trial
        temp_x = np.interp(ref_time,seg_time,xCoord)
        temp_y = np.interp(ref_time,seg_time,yCoord)
        temp_z = np.interp(ref_time,seg_time,zCoord)
    elif condition == "PEACK":
        seg_start = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][1]
        seg_time = all_data_dict["PEACK_time"][participant][trial][seg_start:seg_end]
        # storin the positional coordinates independly 
        xCoord = []; yCoord = []; zCoord = [];
        for coord in all_data_dict["PEACK_positional"][participant][trial][joint][seg_start:seg_end]:
            xCoord.append(coord[0]); yCoord.append(coord[1]); zCoord.append(coord[2])
        # bringing them one the same length as the reference time of the shortest trial
        temp_x = np.interp(ref_time,seg_time,xCoord)
        temp_y = np.interp(ref_time,seg_time,yCoord)
        temp_z = np.interp(ref_time,seg_time,zCoord)

    # applying the trajectory formula
    trajectory_variability = []
    for i in range(0,len(temp_x)):
        trajectory_variability.append(((temp_x[i]-mu_x[i])**2)+((temp_y[i]-mu_y[i])**2)+((temp_z[i]-mu_z[i])**2))
    # returning a trajectory array
    return [trajectory_variability, ref_time]

def calc_acceleration(all_data_dict, participant, trial, mov_seg,joint,joint_selection,condition="VICON", plotting=False):
    if condition == "VICON":
        # determining time of this movement segment
        seg_start = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][1]
        seg_time = all_data_dict["VICON_time"][participant][trial][seg_start:seg_end]
        dt = np.mean(np.diff(seg_time)) # calculating time difference between two points
        acc = np.diff(all_data_dict["VICON_velocity_segmented"][participant][trial][joint][mov_seg]) / dt #acc calc
        if plotting:
            fig, axs = plt.subplots(2)
            fig.suptitle('Velocity vs Acceleration')
            axs[0].plot(seg_time,all_data_dict["VICON_velocity_segmented"][participant][trial][joint][mov_seg])
            axs[1].plot(seg_time[:-1],acc)
            axs[1].hlines(y=0.2, xmin=seg_time[1], xmax=seg_time[-1], linewidth=1, color='r')
            plt.show()
    
    elif condition == "PEACK":
        # determining time of this movement segment
        seg_start = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][1]
        seg_time = all_data_dict["PEACK_time"][participant][trial][seg_start:seg_end]
        dt = np.mean(np.diff(seg_time)) # calculating time difference between two points
        acc = np.diff(all_data_dict["PEACK_velocity_segmented"][participant][trial][joint][mov_seg]) / dt #acc calc
        if plotting:
            fig, axs = plt.subplots(2)
            fig.suptitle('Velocity vs Acceleration')
            axs[0].plot(seg_time,all_data_dict["PEACK_velocity_segmented"][participant][trial][joint][mov_seg])
            axs[1].plot(seg_time[:-1],acc)
            axs[1].hlines(y=0.2, xmin=seg_time[1], xmax=seg_time[-1], linewidth=1, color='r')
            plt.show()
    return acc     # return the acceleration profile

def acc_ratio(all_data_dict, participant, trial, mov_seg,joint,joint_selection,condition = "VICON"):
    acc_mean_peak_ratio = np.mean(calc_acceleration(all_data_dict, participant, trial, mov_seg,joint,joint_selection,condition)) / np.max(calc_acceleration(all_data_dict, participant, trial, mov_seg,joint,joint_selection,condition))
    return acc_mean_peak_ratio

def get_peaks_and_troughs(a, a_time):
    peaks, _ = find_peaks(a) # adapt for distance later
    troughs = []
    for i in range(len(peaks)-1):
        signal_segment = a[peaks[i]:peaks[i+1]]
        trough = peaks[i] + np.argmin(signal_segment)
        troughs.append(trough)
    troughs = np.array(troughs, dtype='int64')
    troughs = np.insert(troughs, 0, 0)
    troughs = np.append(troughs, len(a)-1)
    return [peaks ,troughs]

def num_submov(all_data_dict, participant, trial, mov_seg, joint,joint_selection,condition = "VICON",debug=False):
    if condition == "VICON":
        seg_start = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["VICON_stamps"][participant][trial][joint_selection][mov_seg][1]
        seg_time = all_data_dict["VICON_time"][participant][trial][seg_start:seg_end]
    elif condition == "PEACK":
        seg_start = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][0]
        seg_end = all_data_dict["PEACK_stamps"][participant][trial][joint_selection][mov_seg][1]
        seg_time = all_data_dict["PEACK_time"][participant][trial][seg_start:seg_end]

    [pks, trs] = get_peaks_and_troughs(calc_acceleration(all_data_dict, participant, trial, mov_seg,joint,joint_selection,condition),seg_time[:-1]);
    
    num_submovements = len(pks)
    if debug:
        calc_acceleration(all_data_dict, participant, trial, mov_seg,joint,joint_selection,condition, True)
        for i in pks:
            print(seg_time[i])
    return num_submovements

debug = False
def calc_parameter_matrix(all_data_dict, data_choice, joint_selection):
    num_joints = len(all_data_dict["VICON_filtered"][0][0]); num_mov = var.movNum # hardcoding number of movements and joints
    num_part = var.partNum #len(all_data_dict["PEACK_positional"])
    num_trial = var.maxTrials #len(all_data_dict["PEACK_positional"][participant])
    # create a multi-dimensional matrix containing one's parameters values for all joints, mov. segments, participants and trials
    vicon_ttp = np.zeros([num_joints, num_mov, num_part, num_trial]); vicon_peak = np.zeros([num_joints, num_mov, num_part, num_trial]); vicon_mean_vel = np.zeros([num_joints, num_mov, num_part, num_trial]); vicon_trajec = np.zeros([num_joints, num_mov, num_part, num_trial]); vicon_total_path = np.zeros([num_joints, num_mov, num_part, num_trial]); vicon_acc_rat = np.zeros([num_joints, num_mov, num_part, num_trial]); vicon_submovements = np.zeros([num_joints, num_mov, num_part, num_trial]);
    peack_ttp = np.zeros([num_joints, num_mov, num_part, num_trial]); peack_peak = np.zeros([num_joints, num_mov, num_part, num_trial]); peack_mean_vel = np.zeros([num_joints, num_mov, num_part, num_trial]); peack_trajec = np.zeros([num_joints, num_mov, num_part, num_trial]); peack_total_path = np.zeros([num_joints, num_mov, num_part, num_trial]); peack_acc_rat = np.zeros([num_joints, num_mov, num_part, num_trial]); peack_submovements = np.zeros([num_joints, num_mov, num_part, num_trial]);
    # looping through all sessions to calculate parameters
    for joint in range(num_joints):
        # print("Mov joint", joint)
        for mov_seg in range(num_mov):
            # print("Mov Seg", mov_seg)
            for participant in range(0,len(all_data_dict["PEACK_positional"])):
                # print("Participant", participant)
                for trial in range(0,len(all_data_dict["PEACK_positional"][participant])):
                    # print("Trial", trial)
                    if len(all_data_dict["VICON_stamps"][participant][trial][joint_selection]) == var.movNum and len(all_data_dict["PEACK_stamps"][participant][trial][joint_selection]) == var.movNum:
                        cndn = "VICON"
                        [vicon_ttp[joint][mov_seg][participant][trial], vicon_peak[joint][mov_seg][participant][trial], vicon_mean_vel[joint][mov_seg][participant][trial]] = time_to_peak(all_data_dict, participant, trial, mov_seg,joint, joint_selection,cndn)
                        vicon_total_path[joint][mov_seg][participant][trial] = total_path_calc(all_data_dict, participant, trial, mov_seg,joint,joint_selection,cndn)
                        vicon_acc_rat[joint][mov_seg][participant][trial] = acc_ratio(all_data_dict, participant, trial, mov_seg,joint,joint_selection,cndn)
                        vicon_submovements[joint][mov_seg][participant][trial] = num_submov(all_data_dict, participant, trial, mov_seg,joint,joint_selection,cndn, debug)

                        cndn = "PEACK"
                        [peack_ttp[joint][mov_seg][participant][trial], peack_peak[joint][mov_seg][participant][trial], peack_mean_vel[joint][mov_seg][participant][trial]] = time_to_peak(all_data_dict, participant, trial, mov_seg,joint,joint_selection,cndn)
                        peack_total_path[joint][mov_seg][participant][trial] = total_path_calc(all_data_dict, participant, trial, mov_seg,joint,joint_selection,cndn)
                        peack_acc_rat[joint][mov_seg][participant][trial] = acc_ratio(all_data_dict, participant, trial, mov_seg,joint,joint_selection,cndn)
                        peack_submovements[joint][mov_seg][participant][trial] = num_submov(all_data_dict, participant, trial, mov_seg, joint,joint_selection, cndn, debug)

    # create matrices to store validation metrics
    data_vicon = [vicon_ttp, vicon_peak, vicon_mean_vel, vicon_trajec, vicon_total_path, vicon_acc_rat, vicon_submovements]
    data_peack = [peack_ttp, peack_peak, peack_mean_vel, peack_trajec, peack_total_path, peack_acc_rat, peack_submovements]
    correlation_matrix = np.zeros([num_joints, num_part, num_mov]);
    corr_mat = np.zeros([num_joints, num_mov]);
    # loop through all session to calculate validation score between peack and vicon for single parameters
    for joint in range(num_joints):
        for mov_seg in range(len(data_vicon[data_choice][joint])):
            for participant in range(len(data_vicon[data_choice][joint][mov_seg])):
                if data_choice != 3:
                    a = data_vicon[data_choice][joint][mov_seg][participant]
                    b = data_peack[data_choice][joint][mov_seg][participant]
                    try: 
                        a1=a[a!=0]
                        b1=b[b!=0]
                        pearson = scipy.stats.pearsonr(a1, b1)[0]
                        if np.isnan(pearson):
                            pearson = 0
                    except: 
                        pearson = 0
                    # calculate the difference percentage - pearson may not be applicable in these cases
                    b = np.asarray(b); a = np.asarray(a); 
                    tmp = abs((b-a)/a)
                    tmp=tmp[(1 > tmp)]
                    try:
                        mdp = round(mean(tmp),3)
                    except:
                        mdp = 0
                    correlation_matrix[joint][participant][mov_seg]=1-mdp # alternatively use "round(pearson,3)" to use the pearson validation metric
                # calculate pearson metric for the variability trajectory
                else:
                    tmpPearson = []
                    for trial in range(0,len(all_data_dict["PEACK_positional"][participant])):
                        try:
                            cndn = "VICON"; [a, a_time] = calc_trajectory_var(all_data_dict, participant, trial, mov_seg,joint,joint_selection,cndn)
                            cndn = "PEACK"; [b, b_time] = calc_trajectory_var(all_data_dict, participant, trial, mov_seg,joint,joint_selection,cndn)
                            tmpPearson.append(crosscorr(np.asarray(a),np.asarray(b),np.asarray(a_time),np.asarray(b_time))[0])
                        except:
                            tmpPearson.append(0)
                    if all(v == 0 for v in tmpPearson):
                        pearson = 0
                    else:
                        tmpPearson = np.asarray(tmpPearson)
                        pearson=tmpPearson[tmpPearson!=0].mean()
                    correlation_matrix[joint][participant][mov_seg]=round(pearson,3)
        # take average across participants
        if data_choice == 3:
            for mov_seg in range(num_mov):
                temp=[]
                for participant in range(var.partNum):
                    temp.append(correlation_matrix[joint][participant][mov_seg])
                temp = np.asarray(temp)
                corr_mat[joint][mov_seg]=round(temp[temp>0.5].mean(),3) # everything below 0.5 gets classified as an outlier
        else: 
            corr_mat[joint]=np.mean(correlation_matrix[joint],axis=0)
    
    # OUTPUT
    # print(correlation_matrix)
    with open('x_parameter.pkl', 'wb') as f:
        pickle.dump(correlation_matrix, f) # watch out for other dimension order than velocity profile!
    data_type = ["Time to peak", "Peak Velocity", "Mean Velocity", "Trajectory Variability", "Total Path", "Acceleration Mean / Peak", "Number of Submovements"]
    print(data_type[data_choice])
    print()
    print(corr_mat)
    print()
    return corr_mat

