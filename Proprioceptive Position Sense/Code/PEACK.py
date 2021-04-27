import numpy as np
from common import get_joint


def get_PEACK_UL_prop(pdata, start, samples):

    if samples <= 0:
        data = np.zeros((8, pdata.shape[0]-start+1, 3))
    else:
        data = np.zeros((8, samples, 3))

    start = start - 1
    ref_idx = -1

    joint_idx = 3
    data[0, :, :] = get_joint(pdata, joint_idx, ref_idx, start, samples)

    joint_idx = 4
    data[1, :, :] = get_joint(pdata, joint_idx, ref_idx, start, samples)

    joint_idx = 5
    data[2, :, :] = get_joint(pdata, joint_idx, ref_idx, start, samples)

    joint_idx = 6
    data[3, :, :] = get_joint(pdata, joint_idx, ref_idx, start, samples)

    joint_idx = 7
    data[4, :, :] = get_joint(pdata, joint_idx, ref_idx, start, samples)

    joint_idx = 8
    data[5, :, :] = get_joint(pdata, joint_idx, ref_idx, start, samples)

    joint_idx = 2
    data[6, :, :] = get_joint(pdata, joint_idx, ref_idx, start, samples)

    joint_idx = 9
    data[7, :, :] = get_joint(pdata, joint_idx, ref_idx, start, samples)
    
    return data
