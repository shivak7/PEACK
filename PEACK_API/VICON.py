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

def ProcessAsSingleEpochs(Param):
    print()
    print("Processing Data from: ", Param.DataPath)
    DL = DataLoader(Param.DataPath)
    PEACKData = [];
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
    bar = progressbar.ProgressBar(max_value=len(DL.SubDirs)*DL.SubDirs[0].Nfiles)      #Approximate total number of files
    for i in range(len(DL.SubDirs)):
        tempStruct = []

        for j in range(0,DL.SubDirs[i].Nfiles):
            # if(bar_count == 3):
            #     debug_mode = False
            # else:
            #     debug_mode = False
            tempdata = ExtractKinematicData(DL.getFile(i,j), joint_names, joints, Param.Fs, len(joints), smoothing_alpha = Param.smoothing_alpha, cutoff = Param.lowpass_cutoff, order = Param.lowpass_order, median_filter=Param.median_filter_win, trunc = Param.Trunc, unit_rescale=Param.unit_rescale, type=Param.method, filtered = Param.do_filtering, drop_lower=Param.drop_contiguous_columns, interp_missing = Param.interp_missing, Use2D = Param.Use2D, debug=debug_mode)
            tempStruct.append(tempdata)
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
    #import pdb; pdb.set_trace()

    Results = []
    for i in range(len(TempData)):          #Iterate through tasks
        Participants = []
        for j in range(len(TempData[i])):   #Iterate through participants
            Metrics = []
            # if (j==3):
            #     import pdb; pdb.set_trace()
            print("Processing... ", TempData[i][j].filename)
            for k in range(len(FuncList)):  #Iterate through metric functions to be evaluated
                Metrics.append(FuncList[k](TempData[i][j]))
            Participants.append(Metrics)
        Results.append(Participants)
    return Results
