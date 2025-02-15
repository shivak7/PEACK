import numpy as np

'''
    Copyright 2022 Shivakeshavan Ratnadurai-Giridharan
    A function for selecting Upperlimb timeseries from PEACK kinematic data.

    Update: 10/08/2022
    Changed join index references to start from 0 instead of from 1.
    Neater and now compatible with PartsMap for indexing joints.

    Added 3D/2D option for get_joint (x,y,z vs x,y).

'''

def get_joint(pdata,limb_idx,ref_idx,start,samples, mode='3D'):

    if(mode=='3D'):
        dims = 3
    elif(mode=='2D'):
        dims = 2
    else:
        raise ValueError('get_joint() mode can only be 2D or 3D')

    Data = np.zeros((pdata.shape[0], dims))

    # assigns the adjacent zero array with positonal data of the indv. joint
    if ref_idx <=0:
        if samples <=0:
            Data = pdata[start:, dims*(limb_idx):dims*(limb_idx)+dims]
        else:
            Data = pdata[start:start+samples, dims*(limb_idx):dims*(limb_idx)+dims]
    else:
        if samples <=0:
            Ref = pdata[start:,dims*(ref_idx):dims*(ref_idx)+dims]
            Data = pdata[start:,dims*(limb_idx):dims*(limb_idx)+dims] - Ref
        else:
            Ref = pdata[start:, dims * (ref_idx):dims * (ref_idx) + dims]
            Data = pdata[start:start+samples, dims * (limb_idx):dims * (limb_idx) + dims] - Ref
    return Data


def get_UL_prop(pdata, all_joints, start, samples, n_joints, type, mode='3D'):
    # creates an zero-array with number-of-joints subarrays for positional data later
    if(mode=='3D'):
        dims = 3
    elif(mode=='2D'):
        dims = 2
    else:
        raise ValueError('get_UL_prop() mode can only be 2D or 3D.')

    if samples <= 0:
        data = np.zeros((n_joints, pdata.shape[0]-start+1, dims))
    else:
        data = np.zeros((n_joints, samples, dims))

    start = start - 1; ref_idx = -1

    # loops through all kinematic data to find adjacent joints
    for i in range(0,n_joints):
        data[i, :, :] = get_joint(pdata, all_joints[i], ref_idx, start, samples, mode)

    return data
