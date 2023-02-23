import numpy as np
import os
import pandas as pd
from PreProcess import PreProcess
from joints_from_data import get_UL_prop
import PEACK_Filters as PF
from scipy import stats
import warnings
'''
    Copyright 2023 Jan Breuer & Shivakeshavan Ratnadurai-Giridharan

    A class for loading and processing PEACK / VICON kinematic data.
    Joint data can directly be accessed through Obj<ExtractKinematicData>["joint_name"]

    TODO: Implement case insensitive joint map access
'''

class ExtractKinematicData:

    def __init__(self, fn, joint_names, joints, fs, N_ul_joints, smoothing_alpha=0.5, cutoff=1, order=3, median_filter=1, trunc= [200,0], unit_rescale=1.0, type='PEACK', filtered = False, prior_truncating = True, drop_lower=True, interp_missing = True, Use2D = False, debug = False):
        self.filename = fn
        self.joints = joints
        self.joint_map = dict(zip(joint_names, np.arange(0,N_ul_joints,1)))
        self.type = type
        self.fs = fs
        self.Use2D = Use2D
        self.unit_rescale = unit_rescale
        self.smoothing_alpha = smoothing_alpha
        self.cutoff = cutoff
        self.order = order
        self.median_filter = median_filter
        self.N_ul_joints = N_ul_joints
        self.skiprows = 5 if self.type == 'VICON' else  1                       # Remove! This needs to move to the XML file!!!
        self.data_cols = 60 if self.type == 'VICON' else  76
        self.filtered = filtered
        self.trunc = trunc
        self.prior_truncating = prior_truncating
        self.interp_missing = interp_missing
        self.debug = debug
        self.debug_no = False
        self.getData(drop_lower)

    def getData(self,drop_lower = True):
        # creates a start and end time for a time vector - removes time to avoid edge artifacts
        #temp = pd.read_csv(self.filename,header=None, skiprows=self.skiprows, skip_blank_lines=True).values
        temp = pd.read_csv(self.filename,header=None, index_col=False, names=np.arange(0, self.data_cols,1))

        if(self.skiprows > 0):
            temp = temp.drop(range(0,self.skiprows),axis=0).astype(float).values
        else:
            temp = temp.astype(float).values
        #import pdb; pdb.set_trace()
        start = int((self.trunc[0]/1000)*self.fs)
        stop = int((self.trunc[1]/1000)*self.fs) if int((self.trunc[1]/1000)*self.fs) != 0 else -1

        self.time = temp[start:stop, 0]/self.fs if self.type=='VICON' else temp[start:stop, 0]

        if self.prior_truncating:
            #import pdb; pdb.set_trace()
            data = temp[start:stop, 2:]/self.unit_rescale if self.type=='VICON' else temp[start:stop, 1:]/self.unit_rescale
        else:
            data = temp[:, 2:]/self.unit_rescale if self.type=='VICON' else temp[:, 1:]/self.unit_rescale

        dataMode = '3D'
        if(self.Use2D==True):
            dataMode = '2D'

        data, del_cols = PreProcess(data, dataMode, drop_lower, do_interp=self.interp_missing, debug=self.debug); del temp


        if(len(del_cols)>0):
            all_keys = list(self.joint_map.keys())
            for del_idx in range(len(del_cols)):
                search_idx =  np.argwhere(np.squeeze(self.joints)==del_cols[del_idx])         #Check if invalid columns are in our joint list of interest
                if(len(search_idx)>0):
                    del self.joint_map[all_keys[search_idx[0][0]]]                            #If so, delete the joint-map key/value pair for the invalid joint (nan's)


        # Extract all samples in timeseries
        self.data = get_UL_prop(data, self.joints, 1, -1, self.N_ul_joints, self.type, dataMode)
        if(self.type=='PEACK'):
            dt = np.diff(self.time)
            sdt = (dt==0) + 0.
            t_ratio = np.sum(sdt) / len(dt)

            # only reconstructs it if t_ratio is bigger than 10%
            try:
                true_dt = stats.mode(dt[dt!=0])[0][0]#np.mean(dt[dt!=0])
            except:
                t_ratio = 0
                self.time = start/self.fs + np.arange(0, np.squeeze(self.time).shape[0]/self.fs, 1.0/self.fs)
                dt = np.diff(self.time)
            if(t_ratio>0.1):    #If the timing information is wrong (>10% of times are not updated) then reconstruct time
                self.time = start/self.fs + np.arange(0,len(self.time)*true_dt,true_dt)

        if self.filtered:
            self.data_filtered = np.zeros_like(self.data)
            warnings.filterwarnings("ignore")
            #self.raw_data = np.copy(self.data)
            for j in range(0,self.N_ul_joints):
                joint = np.squeeze(self.data[j, :, :])
                #mf_len = int(self.fs*1.5)
                joint = PF.position_median_filter(joint, self.fs, self.median_filter)
                #joint = PF.exp_smoothing(joint,self.fs, self.smoothing_alpha)
                #import pdb; pdb.set_trace()
                joint = PF.butter_lowpass_filter(joint, self.cutoff, self.fs, self.order)   # Filtering with a 5th order butterworth lowpass filter @ 6Hz.
                self.data_filtered[j, :, :] = joint
        else:
            self.data_filtered=self.data

        if self.prior_truncating == False:
            # this needs to be altered!
            self.data_filtered = self.data_filtered[:,start:-stop,:]

    def reFilter(self,smoothing_alpha, cutoff, order, median_filter, trunc):
        self.smoothing_alpha = smoothing_alpha
        self.cutoff = cutoff
        self.order = order
        self.median_filter = median_filter
        self.trunc = trunc
        self.data_filtered = np.zeros_like(self.data)

        for j in range(0,self.N_ul_joints):
            joint = np.squeeze(self.data[j, :, :])
            mf_len = int(self.fs*1.5)
            joint = PF.position_median_filter(joint, self.fs/self.median_filter)
            joint = PF.exp_smoothing(joint,self.fs, self.smoothing_alpha)
            joint = PF.butter_lowpass_filter(joint, self.cutoff, self.fs, self.order)   # Filtering with a 5th order butterworth lowpass filter @ 6Hz.
            self.data_filtered[j, :, :] = joint

    def __getitem__(self, key):

        try:
            idx = self.joint_map[key]
            if(self.Use2D==True):
                return self.data_filtered[idx, :, :2]
            return self.data_filtered[idx, :, :]
        except KeyError as ke:
            #print("Warning: Key not found: ", key)
            #print("Available keys are: ", list(self.joint_map.keys()))
            return []
