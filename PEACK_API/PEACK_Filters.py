from scipy.signal import butter, lfilter, freqz, filtfilt, medfilt, detrend
from scipy.ndimage import uniform_filter
from matplotlib import pyplot as plt
#from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import numpy as np
'''
    Copyright 2021 Shivakeshavan Ratnadurai-Giridharan

    A set of filters for processing timeseries kinematic data from humans.

'''

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, cutoff, 'low', fs=fs, analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #import pdb; pdb.set_trace()
    mu = np.mean(data)
    y = filtfilt(b, a, data.T - mu, method='gust')#padlen=int(2*fs))
    y = y.T
    return y + mu

def butter_bandpass_filter(data, cutoff, fs, order=3):

    b,a = butter(order, cutoff, 'bandpass', fs=fs)
    y = filtfilt(b, a, data.T, method='gust')#padlen=int(2*fs))
    y = y.T
    return y

def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter(order, cutoff, 'high', fs=fs, analog=False)
    y = filtfilt(b, a, data.T, method='gust')
    return y.T

def position_median_filter(data, fs, mult=1.5):
    mf_len = int(fs*mult)
    zero_buff = np.zeros((int(data.shape[0]/2), data.shape[1]))
    buff_sig = np.vstack((zero_buff,data, zero_buff))

    y1 = medfilt(buff_sig, [mf_len - (1 - mf_len%2),1])
    y2 = medfilt(np.flipud(buff_sig), [mf_len - (1 - mf_len%2),1])
    new_signal = (y1+np.flipud(y2))/2.0
    trunc_signal = new_signal[len(zero_buff):-len(zero_buff),:]
    #import pdb; pdb.set_trace()
    return trunc_signal

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

# def exp_smoothing(data,fs, alpha=0.3):
#     #import pdb; pdb.set_trace()
#     y = np.zeros(data.shape)
#     for i in range(data.shape[1]):
#         y[:,i] = SimpleExpSmoothing(data[:,i]).fit(smoothing_level=alpha,optimized=False).fittedvalues
#     return y

# def exp_smoothing_vector(data,fs, alpha=0.3):
#     #import pdb; pdb.set_trace()
#     y = np.zeros(data.shape)
#     y = SimpleExpSmoothing(data).fit(smoothing_level=alpha,optimized=False).fittedvalues
#     return y


def center_and_clip(data, clip_min = -200, clip_max = 200):

    y = data - np.median(data)
    y = np.clip(y, clip_min, clip_max)
    return y


def sdetrend(data):

    return detrend(data)

def detrend2(data):
    
    x = np.linspace(0,len(data)-1,len(data))
    m = np.mean(np.diff(data))
    c = np.mean(data - m*x)

    return data - (m*x + c)

