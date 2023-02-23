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

        return [np.nanmedian(LElbAngle), np.nanmedian(RElbAngle)]
        #return angle_ratio
    except:
        return np.nan
        return [np.nan, np.nan]
