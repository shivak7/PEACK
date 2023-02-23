from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plt
import PEACK_Filters as PF
from scipy.signal import find_peaks
import variableDeclaration as var

'''
    Copyright 2022 Jan Breuer & Shivakeshavan Ratnadurai-Giridharan

    Functions for segmenting single joint kinematic data into submovements.
    This code extracts velocity with a 6 Hz 5th order lowpass
    filter + median filter with window 1.5 x Fs.
'''

class segment:

    def __init__(self, obj):
        self.obj = obj

    def get_speed(self, LowPassCutOff, LowPassOrder, median_filter_vel, filt_vel= True):   #Takes an ExtractKinematicData object as an argument
        # create a time-difference vector and derive the velocity from the displacement
        dt = np.mean(np.diff(np.squeeze(self.obj.time)))
        self.velocity = np.diff(self.obj.data, axis=1) / dt

        if filt_vel:
            # filters the velocity data
            for j in range(0,self.obj.N_ul_joints):
                joint = np.squeeze(self.velocity[j, :, :])
                if(var.LowPassEnable):
                    joint = PF.butter_lowpass_filter(joint, LowPassCutOff, self.obj.fs, order=LowPassOrder)   # Filtering with a 5th order butterworth lowpass filter @ 6Hz.
                if(var.MedianFilterEnable):
                    joint = PF.velocity_median_filter(joint, self.obj.fs/median_filter_vel)
                #joint = PF.velocity_mean_filter(joint, obj.fs)

                self.velocity[j, :, :] = joint

        self.speed = np.linalg.norm(self.velocity,axis=2)

        return self.speed

    def peaks_and_troughs(a, a_time):
        peaks, _ = find_peaks(a, height=0.2*np.max(a), distance=(round(((1 < a_time) & (a_time < 2)).sum()/10)*10)/(1000.0/400)) # adapt for distance later
        troughs = []
        for i in range(len(peaks)-1):
            signal_segment = a[peaks[i]:peaks[i+1]]
            trough = peaks[i] + np.argmin(signal_segment)
            troughs.append(trough)
        troughs = np.array(troughs, dtype='int64')
        troughs = np.insert(troughs, 0, 0)
        troughs = np.append(troughs, len(a)-1)

        timeStamps = []
        for i in range(len(peaks)):
            #Search left
            left_seg = a[troughs[i]:peaks[i]]; amplitude = a[peaks[i]] - a[troughs[i]]
            left_val = troughs[i] + len(left_seg) - np.argmax(np.flip(left_seg) <= (a[troughs[i]] + (0.1 * amplitude)))
            #Search right
            right_seg = a[peaks[i]:troughs[i+1]]; amplitude = a[peaks[i]] - a[troughs[i+1]]
            right_val = peaks[i] + np.argmax(right_seg <= (a[troughs[i+1]] + (0.1 * amplitude)))
            timeStamps.append([left_val,right_val])
        return np.array(timeStamps)

    def get_speed_filtered(self, LowPassCutOff, LowPassOrder, median_filter_vel):   #Takes an ExtractKinematicData object as an argument
        # create a time-difference vector and derive the velocity from the displacement
        dt = np.mean(np.diff(np.squeeze(self.obj.time)))
        self.velocity = np.diff(self.obj.data_filtered, axis=1) / dt
        #import pdb; pdb.set_trace()
        # filters the velocity data
        for j in range(0,self.obj.N_ul_joints):
            joint = np.squeeze(self.velocity[j, :, :])
            if(var.LowPassEnable):
                joint = PF.butter_lowpass_filter(joint, LowPassCutOff, self.obj.fs, order=LowPassOrder)   # Filtering with a 5th order butterworth lowpass filter @ 6Hz.
            if(var.MedianFilterEnable):
                joint = PF.velocity_median_filter(joint, self.obj.fs/median_filter_vel)
            self.velocity[j, :, :] = joint

        self.speed = np.linalg.norm(self.velocity,axis=2)
        return self.speed

    def get_peaks_and_troughs(self, signal):

        peaks, _ = find_peaks(signal, height=var.peack_detect_threshold_multiplier*np.max(signal), distance=self.obj.fs/(1000.0/var.max_suppression_threshold))
        troughs = []
        for i in range(len(peaks)-1):

            signal_segment = signal[peaks[i]:peaks[i+1]]
            trough = peaks[i] + np.argmin(signal_segment)
            troughs.append(trough)

        troughs = np.array(troughs, dtype='int64')
        troughs = np.insert(troughs, 0, 0)
        troughs = np.append(troughs, len(signal)-1)

        return peaks,troughs

    def get_movement_timestamps(self):
        peaks,troughs = self.get_peaks_and_troughs(self.speed[2,:])
        timeStamps = [];
        for i in range(len(peaks)):

            #Search left
            left_seg = self.speed[2,troughs[i]:peaks[i]]
            amplitude = self.speed[2, peaks[i]] - self.speed[2, troughs[i]]
            left_val = troughs[i] + len(left_seg) - np.argmax(np.flip(left_seg) <= (self.speed[2, troughs[i]] + (0.1 * amplitude)))
            #Search right
            right_seg = self.speed[2,peaks[i]:troughs[i+1]]
            amplitude = self.speed[2, peaks[i]] - self.speed[2, troughs[i+1]]
            right_val = peaks[i] + np.argmax(right_seg <= (self.speed[2, troughs[i+1]] + (0.1 * amplitude)))

            timeStamps.append([left_val,right_val])

        self.timeStamps = np.array(timeStamps)
        return self.timeStamps
