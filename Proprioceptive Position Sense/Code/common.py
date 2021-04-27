import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import directed_hausdorff
import math

def get_joint(pdata,limb_idx,ref_idx,start,samples):

    Data = np.zeros((pdata.shape[0], 3))

    if ref_idx <=0:
        if samples <=0:
            Data = pdata[start:, 3*(limb_idx-1):3*(limb_idx-1)+3]
        else:
            Data = pdata[start:start+samples, 3*(limb_idx-1):3*(limb_idx-1)+3]
    else:
        if samples <=0:
            Ref = pdata[start:,3*(ref_idx-1):3*(ref_idx-1)+3]
            Data = pdata[start:,3*(limb_idx-1):3*(limb_idx-1)+3] - Ref
        else:
            Ref = pdata[start:, 3 * (ref_idx - 1):3 * (ref_idx - 1) + 3]
            Data = pdata[start:start+samples, 3 * (limb_idx - 1):3 * (limb_idx - 1) + 3] - Ref

    return Data


def draw_upper_body(Joints, ax, pltstr):

    right_points = Joints[0:3, :]
    left_points = Joints[3:6, :]
    line = Joints[6:8, :]

    X = np.array(left_points)
    art = ax.plot(X[:,0],X[:,1],X[:,2], linestyle=pltstr, marker='o', color='b')
    X = np.array(right_points)
    ax.plot(X[:,0],X[:,1],X[:,2], linestyle=pltstr, marker='o', color='g')
    X = np.array(line)
    ax.plot(X[:,0],X[:,1],X[:,2], linestyle=pltstr, marker='*', color='k')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    return art


def draw_upper_body_combo(P_Joints, P_X0, V_Joints, V_X0):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    a1 = draw_upper_body(P_Joints,ax,'-')
    #ax.legend(['PEACK with Intel Realsense'])
    a2 = draw_upper_body(V_Joints,ax,'--')
    ax.legend([a1[0], a2[0]],['PEACK with Intel Realsense', 'VICON'])
    plt.xticks(np.arange(-0.2, 0.05, step=0.1))
    ax.set_xlabel('Z (m)')
    ax.set_ylabel('X (m)')
    ax.set_zlabel('Y (m)')

    ax.view_init(15, 160)
    fig.savefig("Alignment" + ".png", dpi=600, bbox_inches='tight')
    plt.show(block=False)


def getNormal(Joints,dev_type):

    X0 = (Joints[0,:] + Joints[3,:])/2.0
    Joints = Joints - X0
    X1 = X0 - X0

    right_points = Joints[0:3, :]
    left_points = Joints[3:6, :]

    if dev_type=='VICON':
        normal = left_points[0,:] - X1
    else:
        normal = left_points[0,:] - X1

    normal = normal/np.sqrt(np.dot(normal,normal.T))

    return X0, Joints, normal

def getNormal2(Joints,dev_type):

    X0 = (Joints[0,:] + Joints[3,:])/2.0
    Joints = Joints - X0
    X1 = X0 - X0
    line = Joints[6:8, :]

    right_points = Joints[0:3, :]
    left_points = Joints[3:6, :]

    if dev_type=='VICON':
        #normal = left_points[0,:] - X0
        v1 = line[0,:] - X1;
        v2 = line[1,:] - X1;
        normal = np.cross(v1,v2)
    else:
        #normal = right_points[0,:] - X0
        v1 = line[0,:] - X1;
        v2 = line[1,:] - X1;
        normal = np.cross(v1,v2)
        # v1 = line[1,:] - right_points[0,:];
        # v2 = line[1,:] - left_points[0,:];
        # normal1 = np.cross(v2,v1)
        # normal1 = normal1/np.sqrt(np.dot(normal1,normal1.T))
        # v3 = line[0,:] - line[1,:]
        # normal = np.cross(v3,normal1)

    #normal = normal/np.sqrt(np.dot(normal,normal.T))
    normal = normal/np.linalg.norm(normal)

    return X0, Joints, normal


def getRotationMatrix(x):

    alpha = x[0]
    beta = x[1]
    gamma = x[2]

    Rx = np.matrix([[1, 0, 0],[0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.matrix([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    #M = np.matmul(Rz,np.matmul(Ry,Rx))
    M = np.matmul(np.matmul(Rz,Ry),Rx)

    return M


def proprioception_cost(X,Y):

    Z = X - Y
    err1 = np.sqrt(np.sum(np.square(Z)))
    #V = X[6:8,:] - Y[6:8,:]
    #err2 = 10000*np.sqrt(np.sum(np.square(V)))/2
    #V1 = np.diff(X[6:8,:],axis=0)
    #V2 = np.diff(Y[6:8,:],axis=0)
    #cosangle = np.arccos(np.abs(np.dot(V1,V2.T)/(np.linalg.norm(V1)*np.linalg.norm(V2))))
    #import pdb; pdb.set_trace()
    return err1


def prop_objective(x,Data):

    M = getRotationMatrix(x)

    P_Joints = Data[0]
    V_Joints = Data[1]

    P_Joints2 = np.matmul(P_Joints,M)
    f = proprioception_cost(P_Joints2[0:8,:], V_Joints[0:8, :])
    return f


def proprioception_align(P_Joints, V_Joints):

    P_X0, P_Joints1, _ = getNormal(P_Joints, 'PEACK')
    V_X0, V_Joints1, _ = getNormal(V_Joints, 'VICON')

    #x0 = np.random.rand(1,3);
    x0 = np.zeros((1,3))
    x =  fmin(prop_objective, x0, args=((P_Joints1, V_Joints1),),full_output=0,disp=0)
    M = getRotationMatrix(x)

    P_Joints2 = np.matmul(P_Joints1, M)

    #draw_upper_body_combo(P_Joints2, P_X0, V_Joints1, V_X0)
    return P_Joints2, V_Joints1, M, (P_X0, V_X0)

def plot_joint_reflection(Joints, reflected, normal):

    plt.rcParams.update({'font.size': 12})
    right_points = Joints[0:3, :]
    left_points = Joints[3:6, :]
    line = Joints[6:8, :]

    X0 = (Joints[0,:] + Joints[3,:])/2.0
    X1 = X0 - X0

    #import pdb; pdb.set_trace()
    xx, yy = np.meshgrid(np.linspace(-0.1,0.1,100), np.linspace(-0.1,0.1,100))
    normal = np.squeeze(normal)
    z = (-normal[0] * xx - normal[1]* yy) * 1. /normal[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.array(left_points)
    a1 = ax.plot(X[:,0],X[:,1],X[:,2], linestyle='-', marker='o', color='b')
    X = np.array(right_points)
    ax.plot(X[:,0],X[:,1],X[:,2], linestyle='-', marker='o', color='g')
    X = np.array(reflected)
    a2 = ax.plot(X[:,0],X[:,1],X[:,2], linestyle='--', marker='o', color='r')
    # X = np.array(line)
    # ax.plot(X[:,0],X[:,1],X[:,2], linestyle='-', marker='*', color='k')

    X = []
    X.append(Joints[0,:])
    X.append(Joints[3,:])
    X = np.array(X)
    ax.plot(X[:,0],X[:,1],X[:,2], linestyle='--', marker='*', color='k')

    ax.plot_surface(xx, yy, z, alpha=0.2)
    plt.xticks(np.arange(-0.2, 0.05, step=0.1))
    ax.set_xlabel('Z (m)')
    ax.set_ylabel('X (m)')
    ax.set_zlabel('Y (m)')
    ax.legend([a1[0], a2[0]],['Actual', 'Reflected'])
    ax.view_init(15, 160)
    #fig.savefig("Reflection_asymmetry" + dev_type + ".png", dpi=600, bbox_inches='tight')
    plt.show()

def Reflection_Error(Joints, normal, X0, dev_type):

    right_points = Joints[0:3, :]
    left_points = Joints[3:6, :]
    line = Joints[6:8, :]
    X0 = X0 - X0;
    d = np.dot(normal,X0.T)
    t = np.dot(right_points,normal.T) - d
    normal = np.resize(normal,(1,3))
    t = np.resize(t,(3,1))

    reflected = right_points - 2 * np.dot(t,normal)
    #plot_joint_reflection(Joints, reflected, normal)
    c1 = np.mean(left_points,axis=0)
    c2 = np.mean(reflected,axis=0)
    err = np.sqrt(np.sum(np.square(c1 - c2)))
    #err = max(directed_hausdorff(left_points, reflected)[0], directed_hausdorff(reflected, left_points)[0])
    #print(np.sum(np.square(c1))/np.sum(np.square(c2)))
    #import pdb; pdb.set_trace()
    return err

def Asymmetry_Error(Joints, dev_type):

    right_points = Joints[1:3, :]
    left_points = Joints[4:6, :]
    line = Joints[6:8, :]

    dist1 = max(directed_hausdorff(left_points, line)[0], directed_hausdorff(line, left_points)[0])
    dist2 = max(directed_hausdorff(right_points, line)[0], directed_hausdorff(line, right_points)[0])
    err = 1 - np.abs((dist1-dist2)/dist1)
    return err

def Asymmetry_Error2(Joints, dev_type):

    right_points = Joints[1:3, :]   # Taking only 2 joints. Ignoring the shoulders
    left_points = Joints[4:6, :]
    line = Joints[6:8, :]

    c_points = (right_points + left_points)/2.0

    r_vecs = right_points - line[1,:]
    l_vecs = left_points - line[1,:]
    rdist = np.linalg.norm(r_vecs,axis=1)
    ldist = np.linalg.norm(l_vecs,axis=1)
    ratios = rdist/ldist
    #err1 = 1 - np.mean(np.abs(1 - ratios))  #np.max is typical
    e = []
    for i in range(len(rdist)):
        den = np.max([rdist[i],ldist[i]])
        num = np.min([rdist[i],ldist[i]])
        e.append((den - num)/den)
    err1 = 1 - np.max(e)
    #import pdb; pdb.set_trace()
    #err1 = np.mean(np.abs(ratios))
    #err1 = 1 - np.abs((1 - err1));

    #import pdb; pdb.set_trace()

    r_vecs = right_points - line[0,:]
    l_vecs = left_points - line[0,:]
    rdist = np.linalg.norm(r_vecs,axis=1)
    ldist = np.linalg.norm(l_vecs,axis=1)
    ratios = rdist/ldist
    #err2 = 1 - np.mean(np.abs(1 - ratios))
    e = []
    for i in range(len(rdist)):
        den = np.max([rdist[i],ldist[i]])
        num = np.min([rdist[i],ldist[i]])
        e.append((den - num)/den)
    err2 = 1 - np.max(e)
    #err2 = np.mean(np.abs(ratios))
    #err2 = 1 - np.abs((1 - err2));
    #import pdb; pdb.set_trace()
    #err = np.max(np.abs(1 - ratios))
    #err = 1 - err
    # if(math.isnan(err1)):
    #     err1 = 1
    #
    # if(math.isnan(err2)):
    #     err2 = 1

    err = np.min([err1,err2])

    #print(dev_type, ratios)
    #err =  np.std(ratios)#np.max(np.abs(1 - rdist/ldist))
    #err = np.linalg.norm(rdist - ldist)
    return err

def proprioception_angles(Joints):

    X0 = (Joints[0,:] + Joints[3,:])/2
    vals = []
    #Rightside
    r_neck_shoulder  = X0 - Joints[0,:]
    r_shoulder_elbow = Joints[1,:] -  Joints[0,:]
    r_elbow_wrist = Joints[2,:] - Joints[1,:]

    v1 = np.matmul(r_neck_shoulder,r_shoulder_elbow.T)
    scale1 = np.linalg.norm(r_neck_shoulder)*np.linalg.norm(r_shoulder_elbow)
    r_sa = np.degrees(np.arccos(v1/scale1)) + np.finfo(float).eps

    v1 = np.matmul(-r_shoulder_elbow,r_elbow_wrist.T)
    scale1 = np.linalg.norm(r_elbow_wrist)*np.linalg.norm(r_shoulder_elbow)
    r_ea = np.degrees(np.arccos(v1/scale1)) + np.finfo(float).eps

    l_neck_shoulder  = X0 - Joints[3,:]
    l_shoulder_elbow = Joints[4,:] -  Joints[3,:]
    l_elbow_wrist = Joints[5,:] - Joints[4,:]

    v1 = np.matmul(l_neck_shoulder,l_shoulder_elbow.T)
    scale1 = np.linalg.norm(l_neck_shoulder)*np.linalg.norm(l_shoulder_elbow)
    l_sa = np.degrees(np.arccos(v1/scale1)) + np.finfo(float).eps

    v1 = np.matmul(-l_shoulder_elbow,l_elbow_wrist.T)
    scale1 = np.linalg.norm(l_elbow_wrist)*np.linalg.norm(l_shoulder_elbow)
    l_ea = np.degrees(np.arccos(v1/scale1)) + np.finfo(float).eps

    vals.append(r_sa); vals.append(r_ea)
    vals.append(l_sa); vals.append(l_ea)
    #err = 1 - (np.abs(1 - l_sa/r_sa) + np.abs(1 - l_ea/r_ea))/2

    # err1 = np.abs(r_ea - l_ea)/180 #1 - (np.abs(1 - l_ea/r_ea) + np.abs(1 - r_ea/l_ea))/2.0
    # err2 = np.abs(r_sa - l_sa)/180#1 - (np.abs(1 - l_sa/r_sa) + np.abs(1 - r_sa/l_sa))/2.0
    err1 = (np.max([r_ea, l_ea]) - np.min([r_ea, l_ea]))/np.max([r_ea, l_ea])
    err2 = (np.max([r_sa, l_sa]) - np.min([r_sa, l_sa]))/np.max([r_sa, l_sa])
    # if(err1>1):
    #     err1 = 1
    # if(err2>1):
    #     err2 = 1

    #err1 = 1 - np.abs(r_ea/l_ea)
    #err2 = 1 - np.abs(r_sa/l_sa)
    #import pdb; pdb.set_trace()
    err1 = 1 - err1;
    err2 = 1 - err2;
    err = np.min([err1,err2])
    #import pdb; pdb.set_trace()
    return err#/2.0#, vals

def dist_traveled(X, Fs):

    dX = np.diff(X,axis=0)
    dX2 = np.square(dX*Fs)
    tX = np.sqrt(np.sum(dX2,axis=1))
    return np.sum(tX[0:])/Fs

def joint_stability(JointTs, Fs=100):

    right_wrist = np.squeeze(JointTs[2,:,:])
    left_wrist = np.squeeze(JointTs[5,:,:])

    r_dt = dist_traveled(right_wrist, Fs)
    l_dt = dist_traveled(left_wrist, Fs)

    #print("Right:", r_dt)
    #print("Left:", l_dt)
    #import pdb; pdb.set_trace()
    #return [r_dt, l_dt]
    return np.max([r_dt, l_dt])

def proprioception_error_combo(P_Joints, V_Joints):


    P_X0, P_Joints1, P_normal = getNormal(P_Joints, 'PEACK')
    V_X0, V_Joints1, V_normal = getNormal(V_Joints, 'VICON')

    PErr = Reflection_Error(P_Joints1, V_normal, P_X0, 'PEACK')
    VErr = Reflection_Error(V_Joints1, V_normal, V_X0, 'VICON')

    norm_angle = np.degrees(np.arccos(np.dot(V_normal, P_normal.T)))
    #print('Norm angle divergence:', norm_angle)
    return PErr, VErr, norm_angle

def proprioception_error_combo_unaligned(P_Joints, V_Joints):

    P_X0, P_Joints1, P_normal = getNormal(P_Joints, 'PEACK')
    V_X0, V_Joints1, V_normal = getNormal(V_Joints, 'VICON')

    PErr = Reflection_Error(P_Joints1, P_normal, P_X0, 'PEACK')
    VErr = Reflection_Error(V_Joints1, V_normal, V_X0, 'VICON')

    # PErr = Asymmetry_Error(P_Joints1, 'PEACK')
    # VErr = Asymmetry_Error(V_Joints1, 'VICON')

    #PErr = Asymmetry_Error2(P_Joints, 'PEACK')
    #VErr = Asymmetry_Error2(V_Joints, 'VICON')

    norm_angle = np.degrees(np.arccos(np.dot(V_normal, P_normal.T)))
    #print('Norm angle divergence:', norm_angle)
    return PErr, VErr, norm_angle
