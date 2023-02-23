'''
    PEACK Proprioception metrics - Shivakeshavan Ratnadurai Giridharan (C) 2023

'''
import numpy as np
from PEACKMetrics import metrics
from PEACKMetrics import graphs
#from PEACKMetrics.metrics

def mirror_symmetry(Body):

    try:
        RWrist = (Body["RWrist1"] + Body["RWrist2"])/2.0
    except ValueError as ve:
        if(len(Body["RWrist1"])>1):
            RWrist = Body["RWrist1"]
        else:
            RWrist = Body["RWrist2"]

    try:
        LWrist = (Body["LWrist1"] + Body["LWrist2"])/2.0
    except ValueError as ve:
        if(len(Body["LWrist1"])>1):
            LWrist = Body["LWrist1"]
        else:
            LWrist = Body["LWrist2"]

    try:
        RElb = (Body["RElbRadial"] + Body["RElbUlnar"])/2.0
    except ValueError as ve:
        if(len(Body["RElbRadial"])>1):
            RElb = Body["RElbRadial"]
        else:
            RElb = Body["RElbUlnar"]

    try:
        LElb = (Body["LElbRadial"] + Body["LElbUlnar"])/2.0
    except ValueError as ve:
        if(len(Body["LElbRadial"])>1):
            LElb = Body["LElbRadial"]
        else:
            LElb = Body["LElbUlnar"]

    v1 = Body["RDeltoid"] - Body["LDeltoid"]
    v2 = RElb - LElb
    v3 = RWrist - LWrist

    cos_theta = (metrics.cosine_between_ts_vectors(v1,v2) + metrics.cosine_between_ts_vectors(v2,v3))/2.0
    mirror_symmetry = (np.pi/4 - np.arccos(cos_theta))/(np.pi/4)
    #Extras = [v2, v3]
    #graphs.Animate(Body, Extras = Extras, plotmode = 'mean')
    #print(np.nanmedian(mirror_symmetry))
    #import pdb; pdb.set_trace()
    return np.nanmedian(mirror_symmetry)


def distance_symmetry_metric_VICON(Body, Ref1="Neck", Ref2="MidSternum"):


    try:
        RWrist = (Body["RWrist1"] + Body["RWrist2"])/2.0
    except ValueError as ve:
        if(len(Body["RWrist1"])>1):
            RWrist = Body["RWrist1"]
        else:
            RWrist = Body["RWrist2"]

    try:
        LWrist = (Body["LWrist1"] + Body["LWrist2"])/2.0
    except ValueError as ve:
        if(len(Body["LWrist1"])>1):
            LWrist = Body["LWrist1"]
        else:
            LWrist = Body["LWrist2"]

    try:
        RElb = (Body["RElbRadial"] + Body["RElbUlnar"])/2.0
    except ValueError as ve:
        if(len(Body["RElbRadial"])>1):
            RElb = Body["RElbRadial"]
        else:
            RElb = Body["RElbUlnar"]

    try:
        LElb = (Body["LElbRadial"] + Body["LElbUlnar"])/2.0
    except ValueError as ve:
        if(len(Body["LElbRadial"])>1):
            LElb = Body["LElbRadial"]
        else:
            LElb = Body["LElbUlnar"]

    if(len(Body["Neck"])>1):
        Neck_Reference = Body["Neck"]
    else:
        Neck_Reference = (Body["RDeltoid"] + Body["LDeltoid"])/2.0

    RWristRef1Vector = RWrist - Neck_Reference
    LWristRef1Vector = LWrist - Neck_Reference
    RElbRef1Vector = RElb - Neck_Reference
    LElbRef1Vector = LElb - Neck_Reference

    RWristRef2Vector = RWrist - Body[Ref2]
    LWristRef2Vector = LWrist - Body[Ref2]
    RElbRef2Vector = RElb - Body[Ref2]
    LElbRef2Vector = LElb - Body[Ref2]

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

def angle_symmetry_metric_VICON(Body):

    MidShldr = (Body["RDeltoid"] + Body["LDeltoid"])/2.0

    try:
        RWrist = (Body["RWrist1"] + Body["RWrist2"])/2.0
    except ValueError as ve:
        if(len(Body["RWrist1"])>1):
            RWrist = Body["RWrist1"]
        else:
            RWrist = Body["RWrist2"]

    try:
        LWrist = (Body["LWrist1"] + Body["LWrist2"])/2.0
    except ValueError as ve:
        if(len(Body["LWrist1"])>1):
            LWrist = Body["LWrist1"]
        else:
            LWrist = Body["LWrist2"]

    try:
        RElb = (Body["RElbRadial"] + Body["RElbUlnar"])/2.0
    except ValueError as ve:
        if(len(Body["RElbRadial"])>1):
            RElb = Body["RElbRadial"]
        else:
            RElb = Body["RElbUlnar"]

    try:
        LElb = (Body["LElbRadial"] + Body["LElbUlnar"])/2.0
    except ValueError as ve:
        if(len(Body["LElbRadial"])>1):
            LElb = Body["LElbRadial"]
        else:
            LElb = Body["LElbUlnar"]

    r_neck_shoulder = Body["RDeltoid"] - MidShldr
    r_shoulder_elbow = RElb - Body["RDeltoid"]
    r_elbow_wrist = RWrist - RElb
    l_neck_shoulder = Body["LDeltoid"] - MidShldr
    l_shoulder_elbow = LElb - Body["LDeltoid"]
    l_elbow_wrist = LWrist - LElb

    RSA = metrics.angle_between_ts_vectors(-r_neck_shoulder, r_shoulder_elbow)
    LSA = metrics.angle_between_ts_vectors(-l_neck_shoulder, l_shoulder_elbow)
    REA = metrics.angle_between_ts_vectors(-r_shoulder_elbow, r_elbow_wrist)
    LEA = metrics.angle_between_ts_vectors(-l_shoulder_elbow, l_elbow_wrist)
    rl_ratio = RSA/LSA
    Angle_metric_shoulders = np.nanmedian(rl_ratio) if np.nanmedian(rl_ratio) < 1 else np.nanmedian(1/rl_ratio)
    rl_ratio = REA/LEA
    Angle_metric_elbows = np.nanmedian(rl_ratio) if np.nanmedian(rl_ratio) < 1 else np.nanmedian(1/rl_ratio)
    return np.min([Angle_metric_shoulders, Angle_metric_elbows])


def orientation_symmetry_metric_VICON(Body):

    try:
        RWrist = (Body["RWrist1"] + Body["RWrist2"])/2.0
    except ValueError as ve:
        if(len(Body["RWrist1"])>1):
            RWrist = Body["RWrist1"]
        else:
            RWrist = Body["RWrist2"]

    try:
        LWrist = (Body["LWrist1"] + Body["LWrist2"])/2.0
    except ValueError as ve:
        if(len(Body["LWrist1"])>1):
            LWrist = Body["LWrist1"]
        else:
            LWrist = Body["LWrist2"]

    try:
        RElb = (Body["RElbRadial"] + Body["RElbUlnar"])/2.0
    except ValueError as ve:
        if(len(Body["RElbRadial"])>1):
            RElb = Body["RElbRadial"]
        else:
            RElb = Body["RElbUlnar"]

    try:
        LElb = (Body["LElbRadial"] + Body["LElbUlnar"])/2.0
    except ValueError as ve:
        if(len(Body["LElbRadial"])>1):
            LElb = Body["LElbRadial"]
        else:
            LElb = Body["LElbUlnar"]

    r_shoulder_elbow = RElb - Body["RDeltoid"]
    r_elbow_wrist = RWrist - RElb
    l_shoulder_elbow = LElb - Body["LDeltoid"]
    l_elbow_wrist = LWrist - LElb

    r_UE_normal = np.cross(-r_shoulder_elbow, r_elbow_wrist)                                #Find normal to plane containing shoulder, elbow and wrist for each side
    l_UE_normal = np.cross(-l_shoulder_elbow, l_elbow_wrist)
    orientation_score = metrics.cosine_between_ts_vectors(r_UE_normal, l_UE_normal)        #sign flipped comparison because shoulder-elbow vectors point in opposite directions
    Extras = [r_UE_normal, l_UE_normal]
    #import pdb; pdb.set_trace()
    print(np.nanmedian(orientation_score))
    if(np.nanmedian(orientation_score) > -0.2):
        graphs.Animate(Body, Extras = Extras, plotmode = 'mean')
        import pdb; pdb.set_trace()
    return np.nanmedian(orientation_score)
