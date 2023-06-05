import numpy as np
from PEACKMetrics.metrics import total_distance_moved, total_angular_distance, angle_between_ts_vectors

def trunk_displacement_angle(Body):

    #Object.data_filtered =
    TrunkVec = Body["Neck"] - Body["MidHip"]
    ShouldersVec = Body["RShoulder"] - Body["LShoulder"]
    #import pdb; pdb.set_trace()
    Angle = angle_between_ts_vectors(TrunkVec, ShouldersVec)

    return np.nanstd(Angle)

def trunk_rotation_angle(Body):

    ThetaSh = total_angular_distance(Body["RShoulder"], Body["LShoulder"])      #Trunk rotation: sum absolute change vs std

    return ThetaSh

def trunk_displacement_distance(Body):

    RSh = total_distance_moved(Body["RShoulder"])
    LSh = total_distance_moved(Body["LShoulder"])

    try:
        Ref = Body["RShoulder"] - Body["RElbow"]
    except:

        try:
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


def elbow_flexion_angle(Body):

    try:

        RShoulderElbow = Body["RShoulder"] - Body["RElbow"]
        LShoulderElbow = Body["LShoulder"] - Body["LElbow"]

        RWristElbow = Body["RWrist"] - Body["RElbow"]
        LWristElbow = Body["LWrist"] - Body["LElbow"]

        RElbAngle = angle_between_ts_vectors(RShoulderElbow, RWristElbow)
        LElbAngle = angle_between_ts_vectors(LShoulderElbow, LWristElbow)

        #ratio = np.nanmedian(RElbAngle / LElbAngle)

        #angle_ratio = ratio if ratio < 1 else 1/ratio
        RElbAngle = np.sort(RElbAngle)
        RElbAngle[np.isnan(RElbAngle)] = []
        LElbAngle = np.sort(LElbAngle)
        LElbAngle[np.isnan(LElbAngle)] = []
        
        l1 = int(len(RElbAngle)*0.05)       # Take top 5 percentile of values
        l2 = int(len(LElbAngle)*0.05)

        
        if(l1 <= 10):
            RElbAngle95 = np.median(RElbAngle)
        else:
            RElbAngle95 = np.median(RElbAngle[-l1:])
        
        if(l2 <= 10):
            LElbAngle95 = np.median(LElbAngle)
        else:
            LElbAngle95 = np.median(LElbAngle[-l1:])

        return LElbAngle95, RElbAngle95
    
    except:
        return np.nan
        return [np.nan, np.nan]
