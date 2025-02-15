import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#from tslearn.utils import  to_time_series_dataset

def time_to_peak(trial):
    return np.argmax(trial)

def peak_value(trial):
    return np.amax(trial)# - np.median(trial) #trial[0]

def smoothness(d_trial, Fs):    #Ensure only 2nd derivative of traj signal is used for this!

    y = np.abs(np.diff(d_trial))
    idx,_ = find_peaks(y, distance=None, height=np.max(y)*0.3)
    # plt.plot(y)
    # plt.plot(idx, y[idx], 'r*')
    # plt.show()
    return len(idx)
    #return np.median(y)
    #return np.std(y)

def path_length(trial):
    return len(trial)

def total_distance_moved(trial):

    centered = trial - np.median(trial,axis=0)
    dtrial = np.diff(centered,axis=0)
    #import pdb; pdb.set_trace()
    return np.sum(np.sqrt(np.sum(dtrial**2,axis=1)))/len(dtrial)                   # Return the avg distance moved across time.


def angular_distance_ts(v1_trial,v2_trial):

    vtrial = v1_trial - v2_trial
    #import pdb; pdb.set_trace()
    vtrial = vtrial / np.linalg.norm(vtrial,axis=1)[:, None]
    ref = vtrial[0]
    Angle = 0
    NanCount = 0
    #import pdb; pdb.set_trace()
    for i in range(1,len(vtrial)):

        theta = np.abs(np.arccos(np.sum(vtrial[i-1]*vtrial[i],axis=-1)))#*(180/np.pi))
        if not np.isnan(theta):
            Angle = Angle + theta
        else:
            NanCount = NanCount + 1
    return Angle/len(vtrial - NanCount)


def cumulative_angular_distance(v1_trial,v2_trial):

    vtrial = v1_trial - v2_trial
    vtrial = vtrial / np.linalg.norm(vtrial,axis=1)[:, None]
    theta = np.zeros(len(vtrial),)
    theta[0] = 0
    for i in range(1,len(theta)):
        theta[i-1] = np.abs(np.arccos(np.sum(vtrial[i-1]*vtrial[i],axis=-1)))
    
    return theta

def total_angular_distance(v1_trial,v2_trial):

    vtrial = v1_trial - v2_trial
    #import pdb; pdb.set_trace()
    vtrial = vtrial / np.linalg.norm(vtrial,axis=1)[:, None]
    ref = vtrial[0]
    Angle = 0
    NanCount = 0
    #import pdb; pdb.set_trace()
    for i in range(1,len(vtrial)):

        theta = np.abs(np.arccos(np.sum(vtrial[i-1]*vtrial[i],axis=-1)))#*(180/np.pi))
        if not np.isnan(theta):
            Angle = Angle + theta
        else:
            NanCount = NanCount + 1
    return Angle/len(vtrial - NanCount)

def cosine_between_ts_vectors(v1,v2):

    dot_prod = np.sum(v1*v2, axis=1)#np.diag(np.dot(v1,v2.T))
    prod_mag = np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)
    
    return dot_prod / prod_mag

def angle_between_ts_vectors(v1,v2):

    cos_angle = cosine_between_ts_vectors(v1, v2)
    angle = np.arccos(cos_angle) + np.finfo(float).eps
    return angle

def angle_between_ts_vector_ref(v1,v2_ref):

    v2 = np.zeros_like(v1) + v2_ref
    cos_angle = cosine_between_ts_vectors(v1, v2)
    angle = np.arccos(cos_angle) + np.finfo(float).eps
    return angle

def classify_clusters(trials, min_monotonic_range=20):

    trend = np.zeros((len(trials),3))
    for i in range(len(trials)):
        for j in range(trials[i].shape[1]):
            trial = trials[i][:,j]
            trial = trial[~np.isnan(trial)]
            d_trend = trial[0] - trial[-1]
            if np.abs(d_trend) <= min_monotonic_range:
                d_trend = 0
            trend[i][j] = np.sign(d_trend)
    #print(trend)
        #plt.plot(trials[i])
        #plt.show()
    bad_idx = np.sign(np.sum(np.abs(trend),axis=1))
    good_idx = np.nonzero(bad_idx)
    bad_idx = np.squeeze(np.nonzero(1 - bad_idx))

    trend = trend[good_idx]


    if(bad_idx.size > 0):
        if(bad_idx.size >1):
            for ele in sorted(np.squeeze(bad_idx), reverse = True):
                del trials[ele]
        else:
                #import pdb; pdb.set_trace()
                del trials[np.squeeze(bad_idx)]

    uv,uc = np.unique(trend, axis=0, return_counts=True)

    labels = -1*np.ones((len(trials),))
    label = 0
    del_list = np.array([],)
    for i in range(len(uv)):

        v = uv[i]
        presence = np.prod(trend == v,axis=1)
        count = np.sum(presence)

        if(count > 2):
            w = np.squeeze(np.where(presence==1))
            labels[w] = label
            label = label + 1
        else:
            w = np.squeeze(np.where(presence==1))
            del_list = np.hstack((del_list,w))

    if(len(del_list)>0):
        for del_iter in sorted(np.int32(del_list), reverse=True):
            del trials[del_iter]
    labels = labels[labels != -1]
    return trials,labels


# def avg_traj(trials, min_monotonic_range = 50, max_length=None):


#     trials,labels = classify_clusters(trials, min_monotonic_range=min_monotonic_range)
#     ts_data = to_time_series_dataset(trials)

#     Ulabels = np.unique(labels)

#     for i in range(len(Ulabels)):
#         X = ts_data[labels==Ulabels[i],:,0]
#         plt.subplot(len(Ulabels),1,i+1)
#         plt.plot(X.T)
#     plt.show()

#     return len(Ulabels)
