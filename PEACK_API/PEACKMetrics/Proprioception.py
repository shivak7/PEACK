'''
    PEACK Proprioception metrics - Shivakeshavan Ratnadurai Giridharan (C) 2023

'''
import numpy as np
from PEACKMetrics import metrics
from PEACKMetrics import graphs
#from PEACKMetrics.metrics

def orientation_symmetry_metric(Body):

    v1 = Body['RShoulder'] - Body['LShoulder']
    v2 = Body['RElbow'] - Body['LElbow']
    v3 = Body['RWrist'] - Body['LWrist']

    #cos_theta = (metrics.cosine_between_ts_vectors(v1,v2) + metrics.cosine_between_ts_vectors(v2,v3))/2.0
    cos_theta1 = np.abs(metrics.cosine_between_ts_vectors(v1,v2))
    cos_theta2 = np.abs(metrics.cosine_between_ts_vectors(v1,v3))
    #mirror_symmetry = (np.pi/4 - np.arccos(cos_theta))/(np.pi/4)
    mirror_symmetry1 = (np.pi/2 - np.arccos(cos_theta1))/(np.pi/2)
    mirror_symmetry2 = (np.pi/2 - np.arccos(cos_theta2))/(np.pi/2)
    #Extras = [v2, v3]
    #graphs.Animate(Body, Extras = Extras, plotmode = 'mean')
    #print(np.nanmedian(mirror_symmetry))
    #import pdb; pdb.set_trace()
    #return np.abs(np.nanmedian(mirror_symmetry))
    return np.min([np.nanmedian(mirror_symmetry1), np.nanmedian(mirror_symmetry2)])


def distance_symmetry_metric(Body):

    Neck_Reference = (Body['RShoulder'] + Body['LShoulder'])/2.0

    
    RWristRef1Vector = Body['RWrist'] - Neck_Reference
    LWristRef1Vector = Body['LWrist'] - Neck_Reference
    RElbRef1Vector = Body['RElbow'] - Neck_Reference
    LElbRef1Vector = Body['LElbow'] - Neck_Reference

    RWristRef2Vector = Body['RWrist'] - Body['Chest']
    LWristRef2Vector = Body['LWrist'] - Body['Chest']
    RElbRef2Vector = Body['RElbow'] - Body['Chest']
    LElbRef2Vector = Body['LElbow'] - Body['Chest']

    RWN = np.linalg.norm(RWristRef1Vector,2,axis=1)     #Right wrist neck distance time-series
    LWN = np.linalg.norm(LWristRef1Vector,2,axis=1)     #Left wrist neck distance time-series
    REN = np.linalg.norm(RElbRef1Vector,2,axis=1)     #Right wrist neck distance time-series
    LEN = np.linalg.norm(LElbRef1Vector,2,axis=1)     #Left wrist neck distance time-series

    RWS = np.linalg.norm(RWristRef2Vector,2,axis=1)     #Right wrist neck distance time-series
    LWS = np.linalg.norm(LWristRef2Vector,2,axis=1)     #Left wrist neck distance time-series
    RES = np.linalg.norm(RElbRef2Vector,2,axis=1)     #Right wrist neck distance time-series
    LES = np.linalg.norm(LElbRef2Vector,2,axis=1)     #Left wrist neck distance time-series


    rl_ratio = RWN/LWN
    Dist_metric_wrist_neck = np.nanmedian(rl_ratio) if np.nanmedian(rl_ratio) < 1 else np.nanmedian(1/rl_ratio)
    rl_ratio = RWS/LWS
    Dist_metric_wrist_sternum = np.nanmedian(rl_ratio) if np.nanmedian(rl_ratio) < 1 else np.nanmedian(1/rl_ratio)
    rl_ratio = REN/LEN
    Dist_metric_elb_neck = np.nanmedian(rl_ratio) if np.nanmedian(rl_ratio) < 1 else np.nanmedian(1/rl_ratio)
    rl_ratio = RES/LES
    Dist_metric_elb_sternum = np.nanmedian(rl_ratio) if np.nanmedian(rl_ratio) < 1 else np.nanmedian(1/rl_ratio)

    return np.min([Dist_metric_elb_sternum, Dist_metric_wrist_sternum, Dist_metric_elb_neck, Dist_metric_wrist_neck])
    #import pdb; pdb.set_trace()



def angle_symmetry_ts(Body):

    # try:
    #     Neck_Reference = Body['Neck']
    # except:
    #     Neck_Reference = (Body['RShoulder'] + Body['LShoulder'])/2.0
    #r_neck_shoulder = Body['RShoulder'] - Neck_Reference
    #l_neck_shoulder = Body['LShoulder'] - Neck_Reference
    
    shoulder_to_shoulder = Body['RShoulder'] - Body['LShoulder']
    r_shoulder_elbow = Body['RElbow'] - Body['RShoulder']
    r_elbow_wrist = Body['RWrist'] - Body['RElbow']
    
    l_shoulder_elbow = Body['LElbow'] - Body["LShoulder"]
    l_elbow_wrist = Body['LWrist'] - Body['LElbow']

    #RSA = metrics.angle_between_ts_vectors(-r_neck_shoulder, r_shoulder_elbow)
    #LSA = metrics.angle_between_ts_vectors(-l_neck_shoulder, l_shoulder_elbow)
    RSA = metrics.angle_between_ts_vectors(-shoulder_to_shoulder, r_shoulder_elbow)
    LSA = metrics.angle_between_ts_vectors(shoulder_to_shoulder, l_shoulder_elbow)
    REA = metrics.angle_between_ts_vectors(-r_shoulder_elbow, r_elbow_wrist)
    LEA = metrics.angle_between_ts_vectors(-l_shoulder_elbow, l_elbow_wrist)

    #angle_symmetry_shoulders = RSA/LSA
    #angle_symmetry_elbows = REA/LEA

    return RSA, LSA, REA, LEA

def angle_symmetry_metric(Body):

    RSA, LSA, REA, LEA = angle_symmetry_ts(Body) 

    rl_ratio = RSA/LSA
    Angle_metric_shoulders = np.nanmedian(rl_ratio) if np.nanmedian(rl_ratio) < 1 else np.nanmedian(1/rl_ratio)
    rl_ratio = REA/LEA
    Angle_metric_elbows = np.nanmedian(rl_ratio) if np.nanmedian(rl_ratio) < 1 else np.nanmedian(1/rl_ratio)
    return np.min([Angle_metric_shoulders, Angle_metric_elbows])
