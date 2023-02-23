from scipy.signal import butter, lfilter, freqz, filtfilt, medfilt
from scipy.ndimage import uniform_filter
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import numpy as np
'''
    Copyright 2021 Shivakeshavan Ratnadurai-Giridharan

    A set of filters for processing timeseries kinematic data from humans.

'''

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #import pdb; pdb.set_trace()
    y = filtfilt(b, a, data.T, method='gust')#padlen=int(2*fs))
    y = y.T
    return y

def position_median_filter(data, fs, mult=1.5):
    mf_len = int(fs*mult)
    y = medfilt(data, [mf_len - (1 - mf_len%2),1])
    return y

def median_filter_vector(data, fs, mult=1.5):

    mf_len = int(fs*mult)
    y = medfilt(data, [mf_len - (1 - mf_len%2)])
    return y

def velocity_median_filter(data, fs, mult=0.3):
    mf_len = int(fs*mult)
    y = medfilt(data, [mf_len - (1 - mf_len%2),1])
    return y

def velocity_mean_filter(data, fs, mult=0.6):
    mf_len = int(fs*mult)
    y = uniform_filter(data, [mf_len,1])
    return y

def exp_smoothing(data,fs, alpha=0.3):
    #import pdb; pdb.set_trace()
    y = np.zeros(data.shape)
    for i in range(data.shape[1]):
        y[:,i] = SimpleExpSmoothing(data[:,i]).fit(smoothing_level=alpha,optimized=False).fittedvalues
    return y

def exp_smoothing_vector(data,fs, alpha=0.3):
    #import pdb; pdb.set_trace()
    y = np.zeros(data.shape)
    y = SimpleExpSmoothing(data).fit(smoothing_level=alpha,optimized=False).fittedvalues
    return y


def center_and_clip(data, clip_min = -200, clip_max = 200):

    y = data - np.median(data);
    y = np.clip(y, clip_min, clip_max)
    return y
