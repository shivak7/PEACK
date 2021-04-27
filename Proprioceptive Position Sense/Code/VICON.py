import numpy as np
from common import get_joint


def get_VICON_UL_prop(vdata, start, samples):

    if samples <= 0:
        data = np.zeros((8, vdata.shape[0]-start+1, 3))
    else:
        data = np.zeros((8, samples, 3))

    ref_idx = -1
    start = start - 1

    joint_idx = 1
    data[0, :, :] = get_joint(vdata, joint_idx, ref_idx, start, samples)

    joint_idx = 3
    t1 = get_joint(vdata, joint_idx, ref_idx, start, samples)
    t2 = get_joint(vdata, joint_idx+1, ref_idx, start, samples)
    data[1, :, :] = (t1 + t2)/2.0

    joint_idx = 6
    data[2, :, :] = get_joint(vdata, joint_idx, ref_idx, start, samples)

    joint_idx = 9
    data[3, :, :] = get_joint(vdata, joint_idx, ref_idx, start, samples)

    joint_idx = 11
    t1 = get_joint(vdata, joint_idx, ref_idx, start, samples)
    t2 = get_joint(vdata, joint_idx+1, ref_idx, start, samples)
    data[4, :, :] = (t1 + t2)/2.0

    joint_idx = 14
    data[5, :, :] = get_joint(vdata, joint_idx, ref_idx, start, samples)

    joint_idx = 18
    data[6, :, :] = get_joint(vdata, joint_idx, ref_idx, start, samples)

    joint_idx = 19
    data[7, :, :] = get_joint(vdata, joint_idx, ref_idx, start, samples)

    return data
