from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np


def plot_ts(X, Y = None):

    if Y is None:
        plt.plot(X)
    else:
        plt.plot(X,Y)

    plt.show()


def plot3Dline(ax,pt1,pt2,idx,plotmode = 'default', color = 'gray'):

    X = [pt1[:,0], pt2[:,0]]
    Y = [pt1[:,1], pt2[:,1]]
    Z = [pt1[:,2], pt2[:,2]]

    if plotmode == "default":
        #import pdb; pdb.set_trace()
        X = [pt1[idx][0], pt2[idx][0]]
        Y = [pt1[idx][1], pt2[idx][1]]
        Z = [pt1[idx][2], pt2[idx][2]]

        ax.plot(X, Y, Z, color)
        return

    if plotmode == "mean":
        fn = np.mean
    elif plotmode == "median":
        fn = np.median

    #import pdb; pdb.set_trace()

    X[0] = fn(X[0])
    X[1] = fn(X[1])
    Y[0] = fn(Y[0])
    Y[1] = fn(Y[1])
    Z[0] = fn(Z[0])
    Z[1] = fn(Z[1])

    ax.plot(X, Y, Z, color)


def Animate(Body, idx = 0, Extras = [], plotmode = 'default'):

    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = plt.axes(projection="3d")
    ax.set_proj_type('ortho')

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

    # Draw Trunk
    idx = 0#np.arange(0,len(MidShldr))
    #import pdb; pdb.set_trace()
    #plot3Dline(ax, Body["Neck"], MidShldr, idx)
    plot3Dline(ax, Body["RDeltoid"], MidShldr, idx, plotmode)
    plot3Dline(ax, Body["LDeltoid"], MidShldr, idx, plotmode)
    plot3Dline(ax, MidShldr, Body["MidSternum"], idx, plotmode)

    #Draw upper limbs
    plot3Dline(ax, Body["RDeltoid"], RElb, idx, plotmode)
    plot3Dline(ax, RElb, RWrist, idx, plotmode)
    plot3Dline(ax, Body["LDeltoid"], LElb, idx, plotmode)
    plot3Dline(ax, LElb, LWrist, idx, plotmode)


    #pt1 = np.median(RElb, axis=0)
    pt1 = np.array([0,0,0]).reshape(-1,3)
    # plot3Dline(ax, pt1 + np.array([0,0,0]).reshape(-1,3), pt1+np.array([0,0,1]).reshape(-1,3), idx, plotmode, 'red')
    # plot3Dline(ax, pt1 + np.array([0,0,0]).reshape(-1,3), pt1+np.array([0,1,0]).reshape(-1,3), idx, plotmode, 'red')
    # plot3Dline(ax, pt1 + np.array([0,0,0]).reshape(-1,3), pt1+np.array([1,0,0]).reshape(-1,3), idx, plotmode, 'red')
    # plot3Dline(ax, pt1, RElb - Body["RDeltoid"], idx, plotmode)
    # plot3Dline(ax, pt1, RElb - RWrist, idx, plotmode)


    cols = ['red', 'blue']
    if(len(Extras)>0):
        ExStart = [pt1, pt1]
        for i in range(len(Extras)):
            plot3Dline(ax,ExStart[i],Extras[i], 0, plotmode,cols[i])

    #ax.set_ylim(0, 2)  #X axis
    #ax.set_xlim(0.5, 1)
    #ax.set_zlim(0, 1.5)  #Y axis

    plt.show()
