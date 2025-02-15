import numpy as np
import os
import pandas as pd
from PreProcess import PreProcess
from joints_from_data import get_UL_prop
import PEACK_Filters as PF
from scipy import stats
from scipy.interpolate import CubicSpline
import warnings
from matplotlib import pyplot as plt
'''
    Copyright 2024 Shivakeshavan Ratnadurai-Giridharan

    A class for loading and processing PEACK / VICON kinematic data.
    Joint data can directly be accessed through Obj<ExtractKinematicData>["joint_name"]

    TODO: Implement case insensitive joint map access
'''

class ExtractKinematicData:

    def __init__(self, fn, joint_names, joints, fs, N_ul_joints, smoothing_alpha=0.5, cutoff=1, order=3, median_filter=1, trunc= [200,0], unit_rescale=1.0, type='PEACK', filtered = False, prior_truncating = True, drop_lower=0, interp_missing = True, zeros_as_nan=False, Use2D = False, time_unit='s',debug = False):
        self.filename = fn
        self.joints = joints
        self.joint_map = dict(zip(joint_names, np.arange(0,N_ul_joints,1)))
        self.type = type
        self.fs = np.abs(fs)
        self.var_frame_rate = np.sign(fs)==-1
        self.Use2D = Use2D
        self.zeros_as_nan = zeros_as_nan
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
        self.time_unit = time_unit
        self.debug = debug
        self.debug_no = False
        self.get_filtered_data_by_default = True                       # Default data accessed will be filtered
        self.getData(drop_lower)

    def getData(self,drop_lower = True):
        # creates a start and end time for a time vector - removes time to avoid edge artifacts
        #temp = pd.read_csv(self.filename,header=None, skiprows=self.skiprows, skip_blank_lines=True).values
        #print(self.filename)
        temp = pd.read_csv(self.filename,header=None, index_col=False)#, names=np.arange(0, self.data_cols,1))

        if(self.skiprows > 0):
            temp = temp.drop(range(0,self.skiprows),axis=0).astype(float).values
        else:
            temp = temp.astype(float).values
        
        t_scale = 1
        if self.time_unit=='ms':
            t_scale = 1000
        
        dataMode = '3D'
        if(self.Use2D==True):
            dataMode = '2D'
        
        start = int((self.trunc[0])*self.fs)
        stop = int((self.trunc[1])*self.fs) if int((self.trunc[1])*self.fs) != 0 else -1
        #import pdb; pdb.set_trace()
        self.time = temp[:, 0]/t_scale#/self.fs if self.type=='VICON' else temp[start:stop, 0]/t_scale
        t0 = temp[:, 0]/t_scale

        data = temp[:, 2:] if self.type=='VICON' else temp[:, 1:]

        data, del_cols = PreProcess(data, dataMode, drop_lower, do_interp=self.interp_missing, interp_limit= len(data)/3, zeros_as_nan=self.zeros_as_nan, debug=self.debug); del temp
        
        if self.var_frame_rate==True:
            data, self.time = self.resampleData(data, t0, self.fs)


        #import pdb; pdb.set_trace()
        
        if self.prior_truncating:
            data = data[start:stop, :]/self.unit_rescale
            self.time = self.time[start:stop]
                
        
        if(len(del_cols)>0):
            all_keys = list(self.joint_map.keys())
            for del_idx in range(len(del_cols)):
                search_idx =  np.argwhere(np.squeeze(self.joints)==del_cols[del_idx])         #Check if invalid columns are in our joint list of interest
                if(len(search_idx)>0):
                    del self.joint_map[all_keys[search_idx[0][0]]]                            #If so, delete the joint-map key/value pair for the invalid joint (nan's)


        # Extract all samples in timeseries
        self.data = get_UL_prop(data, self.joints, 1, -1, self.N_ul_joints, self.type, dataMode)
        
        
        # try:

        #     assert self.data.shape[1] == len(self.time)
        # except:
        #     import pdb; pdb.set_trace()
        
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
            if(t_ratio>0.05):    #If the timing information is wrong (>10% of times are not updated) then reconstruct time
                self.time = start/self.fs + np.arange(0,len(self.time)*true_dt,true_dt)
            elif t_ratio > 0:
                t_idx = np.argwhere(dt==0)
                t_idx += 1
                for iter_t in t_idx:
                    if(iter_t < len(self.time)-1):
                        self.time[iter_t] = (self.time[iter_t - 1] + self.time[iter_t + 1])/2
                    else:
                        self.time[iter_t] = self.time[iter_t] + (1/self.fs)
                #import pdb; pdb.set_trace()

        if self.filtered:
            self.data_filtered = np.zeros_like(self.data)
            warnings.filterwarnings("ignore")
            #self.raw_data = np.copy(self.data)
            for j in range(0,self.N_ul_joints):
                #import pdb; pdb.set_trace()
                joint = np.squeeze(self.data[j, :, :])
                joint = PF.position_median_filter(joint, self.fs, self.median_filter)
                #joint = PF.butter_highpass_filter(joint, 0.2, self.fs, 5)
                if not np.isnan(joint).any():
                    #import pdb; pdb.set_trace ()
                    joint = PF.butter_lowpass_filter(joint, self.cutoff, self.fs, self.order)   # Filtering with a 5th order butterworth lowpass filter @ 6Hz.
                else:
                    print('Discountinuous/Missing data, cannot filter joint', j, 'in file:', self.filename)
                self.data_filtered[j, :, :] = joint
        else:
            self.data_filtered=self.data
        
        if self.debug:
            import pdb; pdb.set_trace()
        #plt.plot(self.time, self["RWrist"]); plt.show()
        #import pdb; pdb.set_trace()
        # if self.prior_truncating == False:
        #     # this needs to be altered!
        #     self.data_filtered = self.data_filtered[:,start:-stop,:]

    def resampleData(self, X, t_vec_original, fs):

        t_vec_desired = np.linspace(t_vec_original[0], t_vec_original[-1], int((t_vec_original[-1] - t_vec_original[0])*fs))
        data_original = X#X[:, 2:] if self.type=='VICON' else X[:, 1:]
        data_resampled = np.zeros((len(t_vec_desired), data_original.shape[1]))

        for idx in range(data_original.shape[1]):
            y = data_original[:,idx]
            spl_sig1 = CubicSpline(t_vec_original, y)
            sig1_re = spl_sig1(t_vec_desired)
            data_resampled[:, idx] = sig1_re
        
        #import pdb; pdb.set_trace()
        return data_resampled, t_vec_desired
        

        

    def reFilter(self,smoothing_alpha, cutoff, order, median_filter, trunc):
        
        print("ExtractKinematicData member function reFilter being called.")
        self.smoothing_alpha = smoothing_alpha
        self.cutoff = cutoff
        self.order = order
        self.median_filter = median_filter
        self.trunc = trunc
        self.data_filtered = np.zeros_like(self.data)

        for j in range(0,self.N_ul_joints):
            joint = np.squeeze(self.data[j, :, :])
            joint = PF.position_median_filter(joint, self.fs, self.median_filter)
            #joint = PF.exp_smoothing(joint,self.fs, self.smoothing_alpha)
            joint = PF.butter_lowpass_filter(joint, self.cutoff, self.fs, self.order)   # Filtering with a 5th order butterworth lowpass filter @ 6Hz.
            self.data_filtered[j, :, :] = joint


    def __getitem__(self, key):

        try:
            idx = self.joint_map[key]
        except KeyError as ke:
            #print("Warning: Key not found: ", key)
            #print("Available keys are: ", list(self.joint_map.keys()))
            return []      
        if self.get_filtered_data_by_default == True:
            if(self.Use2D==True):
                    return self.data_filtered[idx, :, :2]
            return self.data_filtered[idx, :, :]
        else:
            #import pdb; pdb.set_trace()
            if(self.Use2D==True):
                    return self.data[idx, :, :2]
            return self.data[idx, :, :]

    
    def __setitem__(self, keys, val):
        
        try:
            idx = self.joint_map[keys[0]]
            self.joint_map[keys[1]] = self.joint_map[keys[0]]
            
            if self.get_filtered_data_by_default == True:
                if(self.Use2D==True):
                    self.data_filtered[idx, :, :2] = val
                self.data_filtered[idx, :, :] = val
            else:
                if(self.Use2D==True):
                    self.data[idx, :, :2] = val
                self.data[idx, :, :] = val
            
        except KeyError as ke:
            newidx = np.max(list(self.joint_map.values())) + 1
            self.joint_map[keys[1]] = newidx
            if self.get_filtered_data_by_default == True:
                self.data_filtered = np.append(self.data_filtered, val[np.newaxis,:], axis=0)
            else:
                self.data = np.append(self.data, val[np.newaxis,:], axis=0)
            #import pdb; pdb.set_trace()
        

    def __delitem__(self, key):
        
        try:
            del self.joint_map[key]
        except:
            return []

    def swapKeys(self, old_key, new_key):
        
        try:
            self.joint_map[new_key] = self.joint_map[old_key]
        except KeyError as ke:
            print('Swapping keys: Key name mismatch!')
            import pdb; pdb.set_trace()
            return False
        
        del self.joint_map[old_key]
        return True
        
        
