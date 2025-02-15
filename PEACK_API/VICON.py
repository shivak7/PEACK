import os
import scipy
import numpy as np
import sys
import pickle
import time
from DataLoader import DataLoader
from ExtractKinematicData import ExtractKinematicData
from OP_maps import PartsMap
from TaskTimes import TaskTimes
import matplotlib.pyplot as plt
import progressbar
import pandas as pd
import re

def VICON_joint_remapper(ViconBody):

    bool_array = [True, False]
    for i in range(2):

        ViconBody.get_filtered_data_by_default = bool_array[i]
        try:
            RWrist = (ViconBody["RWrist1"] + ViconBody["RWrist2"])/2.0
            ViconBody['RWrist1', 'RWrist'] = RWrist
        except ValueError as ve:
            if(len(ViconBody["RWrist1"])>1):
                RWrist = ViconBody["RWrist1"]
                ViconBody['RWrist1', 'RWrist'] = RWrist
            else:
                RWrist = ViconBody["RWrist2"]
                ViconBody['RWrist2', 'RWrist'] = RWrist

        try:
            LWrist = (ViconBody["LWrist1"] + ViconBody["LWrist2"])/2.0
            ViconBody['LWrist1', 'LWrist'] = LWrist
        except ValueError as ve:
            if(len(ViconBody["LWrist1"])>1):
                LWrist = ViconBody["LWrist1"]
                ViconBody['LWrist1', 'LWrist'] = LWrist
            else:
                LWrist = ViconBody["LWrist2"]
                ViconBody['LWrist2', 'LWrist'] = LWrist
        try:
            import pdb; pdb.set_trace()
            RElb = (ViconBody["RElbRadial"] + ViconBody["RElbUlnar"])/2.0
            ViconBody['RElbRadial', 'RElbow'] = RElb
        except ValueError as ve:
            if(len(ViconBody["RElbRadial"])>1):
                RElb = ViconBody["RElbRadial"]
                ViconBody['RElbRadial', 'RElbow'] = RElb
            else:
                RElb = ViconBody["RElbUlnar"]
                ViconBody['RElbUlnar', 'RElbow'] = RElb

        try:
            LElb = (ViconBody["LElbRadial"] + ViconBody["LElbUlnar"])/2.0
            ViconBody['LElbRadial', 'LElbow'] = LElb
        except ValueError as ve:
            if(len(ViconBody["LElbRadial"])>1):
                LElb = ViconBody["LElbRadial"]
                ViconBody['LElbRadial', 'LElbow'] = LElb
            else:
                LElb = ViconBody["LElbUlnar"]
                ViconBody['LElbUlnar', 'LElbow'] = LElb

        #ViconBody['LElbRadial', 'LElbow'] = LElb
        #ViconBody['RElbRadial', 'RElbow'] = RElb
        #ViconBody['LWrist1', 'LWrist'] = LWrist
        #ViconBody['RWrist1', 'RWrist'] = RWrist
    #import pdb; pdb.set_trace()
    
    ViconBody.get_filtered_data_by_default = True
    ViconBody.swapKeys('LDeltoid', 'LShoulder')
    ViconBody.swapKeys('RDeltoid', 'RShoulder')
    ViconBody.swapKeys('MidSternum', 'Chest')
    

    del ViconBody['RElbRadial']; del ViconBody['LElbRadial']
    del ViconBody['RElbUlnar']; del ViconBody['LElbUlnar']
    del ViconBody['RWrist1']; del ViconBody['LWrist1']
    del ViconBody['RWrist2']; del ViconBody['LWrist2']
    return ViconBody

def ProcessAsSingleEpochs(Param, remapFunction=VICON_joint_remapper):
    print()
    print("Processing Data from: ", Param.DataPath)
    DL = DataLoader(Param.DataPath)
    PEACKData = []
    Map = PartsMap(Param.MapFile)
    joints = []
    joint_names = []
    bar_count = 0
    debug_mode = False
    for str in Param.JointNames:
        try:
            joint_names.append(str)
            joints.append(Map[str])
        except:
            print("Error! Key: ", str, " not found in mapping list:\n")
            print(Map)
            raise SystemExit
    bar = progressbar.ProgressBar(max_value=len(DL.SubDirs)*DL.SubDirs[0].Nfiles + 1)      #Approximate total number of files
    for i in range(DL.NsubDirs):
        tempStruct = []

        for j in range(0,DL.SubDirs[i].Nfiles):
            # if(bar_count == 3):
            #     debug_mode = False
            # else:
            #     debug_mode = False
            tempdata = ExtractKinematicData(fn=DL.getFile(i,j), 
                                            joint_names = joint_names, 
                                            joints = joints, 
                                            fs = Param.Fs, 
                                            N_ul_joints = len(joints), 
                                            smoothing_alpha = Param.smoothing_alpha, 
                                            cutoff = Param.lowpass_cutoff, 
                                            order = Param.lowpass_order, 
                                            median_filter=Param.median_filter_win, 
                                            trunc = Param.Trunc, 
                                            unit_rescale=Param.unit_rescale, 
                                            type=Param.method, 
                                            filtered = Param.do_filtering, 
                                            drop_lower=Param.drop_contiguous_columns, 
                                            interp_missing = Param.interp_missing, 
                                            Use2D = Param.Use2D, 
                                            debug=debug_mode)
            if remapFunction:
                Body = remapFunction(tempdata)
            else:
                Body = tempdata
            #import pdb; pdb.set_trace()
            tempStruct.append(Body)
            bar.update(bar_count)
            bar_count = bar_count + 1

        PEACKData.append(tempStruct)

    progressbar.streams.flush()
    progressbar.streams.wrap_stdout()
    redirect_stdout=True
    bar.finish()
    with open(Param.OutFile, 'wb') as f:
        pickle.dump(PEACKData, f)


def plot_curve(y,x=[]):
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    if(x!=[]):
        ax.plot(x,y)
    else:
        ax.plot(y)
    plt.show(block=True)


def AnalyzeAsSingleEpochs(Param, FuncList=[]):

    with open(Param.OutFile, 'rb') as f:
        TempData = pickle.load(f)
    
    len_groups = len(TempData)
    Results = []
    SubIDs = []
    bar_count = 0
    approx_data_len = len(TempData) * len(TempData[0]) + 1
    bar = progressbar.ProgressBar(max_value=approx_data_len)      #Approximate total number of files
    for i in range(len_groups):          #Iterate through tasks
        Participants = []
        PID = []

        n_files = len(TempData[i])
        for j in range(n_files):   #Iterate through participants
            fn = os.path.split(TempData[i][j].filename)[1]
            fn = fn.replace('-','_')
            fn = fn.replace('.','_')
            sID = fn.split('_')[0]
            #import pdb; pdb.set_trace()        
            Metrics = []
            # if (j==3):
            #     import pdb; pdb.set_trace()
            #print("Processing... ", TempData[i][j].filename)
            for k in range(len(FuncList)):  #Iterate through metric functions to be evaluated
                Metrics.append(FuncList[k](TempData[i][j]))
            Participants.append(np.hstack(Metrics))
            PID.append(sID)
            bar.update(bar_count)
            bar_count = bar_count + 1

        Results.append(Participants)
        SubIDs.append(PID)

    progressbar.streams.flush()
    progressbar.streams.wrap_stdout()
    redirect_stdout=False
    bar.finish()
    return Results, SubIDs
    
