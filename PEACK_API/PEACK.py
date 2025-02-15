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
from difflib import get_close_matches
import progressbar
import re

def ProcessAsMultiEpochs(Param):

    DL = DataLoader(Param.DataPath)
    PEACKData = []
    Map = PartsMap(Param.MapFile)
    joints = []
    joint_names = []
    for str in Param.JointNames:
        try:
            joint_names.append(str)
            joints.append(Map[str])
        except:
            print("Error! Key: ", str, " not found in mapping list:\n")
            print(Map)
            raise SystemExit

    Times = TaskTimes(Param.TimeSegFile)     #Load csv file containing task-segmented timestamps

    for j in range(0,DL.SubDirs[0].Nfiles):
        
        print("File: ", j)
        TimeKeyAll = list(Times)
        #FileString = DL.SubDirs[0].Files[j][0:8].replace(' ','')               # This is how it was originally for the AHA
        FileString = DL.SubDirs[0].Files[j].replace(' ','')
        TimeKey = get_close_matches(FileString,TimeKeyAll, cutoff=0.3)                       # Find closest match to subject key in task-time tables
        num_match_flag = False
        #import pdb; pdb.set_trace()
        if len(TimeKey) > 0:
            TimeKey = TimeKey[0]                                                 # Need only closest match from the list
            num_match_flag = re.findall(r'\d+', TimeKey)[0] == re.findall(r'\d+', FileString)[0]    #Check if the exact numerals match between the candidate key and filename
        #import pdb; pdb.set_trace()
        if num_match_flag == True:                                                #If timesegment subject name matches with subject's kinematic data filename, they are a match!
            TaskStart = Times[TimeKey][0::2]
            TaskEnd = Times[TimeKey][1::2]
        else:
            print("Error! Missing file for: ", FileString, " OR Task Time Segmented Data not in sorted order!")
            import pdb; pdb.set_trace()
            raise SystemExit
        tempStruct = []
        for t in range(len(TaskStart)):

            start = TaskStart[t]
            stop = TaskEnd[t]
            #import pdb; pdb.set_trace()
            if np.isnan(start) or np.isnan(stop):
                tempdata = np.nan
            #import pdb; pdb.set_trace()
            else:
                tempdata = ExtractKinematicData(fn=DL.getFile(0,j), 
                                                joint_names = joint_names, 
                                                joints = joints, 
                                                fs = Param.Fs, 
                                                N_ul_joints=len(joints), 
                                                smoothing_alpha = Param.smoothing_alpha, 
                                                cutoff = Param.lowpass_cutoff, 
                                                order = Param.lowpass_order, 
                                                median_filter=Param.median_filter_win, 
                                                trunc= [start,stop], 
                                                unit_rescale=Param.unit_rescale, 
                                                type=Param.method, 
                                                filtered = Param.do_filtering, 
                                                drop_lower=Param.drop_contiguous_columns,
                                                interp_missing=Param.interp_missing, 
                                                zeros_as_nan=Param.zeros_as_nan,
                                                Use2D = Param.Use2D,
                                                time_unit = Param.time_unit)

            tempStruct.append(tempdata)
            print("Task ", t, " Kinematic Data Extracted")
        PEACKData.append(tempStruct)
        with open(Param.OutFile, 'wb') as f:
            pickle.dump(PEACKData, f)



def AnalyzeEpochs(Param, FuncList=[], ArgList=[]):

    with open(Param.OutFile, 'rb') as f:
        TempData = pickle.load(f)
    #import pdb; pdb.set_trace()
    Results = []
    SourceLabel = []
    for i in range(len(TempData)):          #Iterate through participants
        Participants = []
        fname = os.path.split(TempData[i][0].filename)[1]
        fname = fname.replace('-','_')
        fname = fname.replace('.','_')
        subname = re.split('_', fname)[0]
        for j in range(len(TempData[i])):   #Iterate through trials
            Metrics = []
            for k in range(len(FuncList)):  #Iterate through metric functions to be evaluated
                if len(ArgList) > 0:
                    Metrics.append(FuncList[k](TempData[i][j], **ArgList[k]))
                else:
                    Metrics.append(FuncList[k](TempData[i][j]))
            Participants.append(np.hstack(Metrics))
        Results.append(Participants)
        SourceLabel.append(subname)
    #import pdb; pdb.set_trace()
    return Results, SourceLabel

def ProcessAsSingleEpochs(Param):
    print()
    print("Processing Data from: ", Param.DataPath)
    DL = DataLoader(Param.DataPath)
    #import pdb; pdb.set_trace()
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
            # if(bar_count == 10):
            #     debug_mode = True
            # else:
            #     debug_mode = False
            Body = ExtractKinematicData(DL.getFile(i,j), 
                                        joint_names, joints, 
                                        Param.Fs, 
                                        len(joints), 
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
                                        debug=debug_mode,
                                        time_unit = Param.time_unit)
            #Body = VICON_joint_remapper(tempdata)
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

def AnalyzeAsSingleEpochs(Param, FuncList=[], ArgList=[]):

    with open(Param.OutFile, 'rb') as f:
        TempData = pickle.load(f)
    
    Results = []
    SubIDs = []
    bar_count = 0
    

    len_groups = len(TempData)

    len_files = len(TempData[0])
    
    approx_data_len = len_groups * len_files + 1
    bar = progressbar.ProgressBar(max_value=approx_data_len)      #Approximate total number of files
    for i in range(len_groups):          #Iterate through tasks
        Participants = []
        PID = []

        n_files = len(TempData[i])
       
        for j in range(n_files):   #Iterate through participants
            
            Obj = TempData[i][j]

            fn = os.path.split(Obj.filename)[1]
            fn = fn.replace('-','_')
            fn = fn.replace('.','_')
            sID = fn.split('_')[0]
            sID = sID.split('/')[-1]
            sID = sID.split('\\')[-1]
            #import pdb; pdb.set_trace()        
            Metrics = []
            
            for k in range(len(FuncList)):  #Iterate through metric functions to be evaluated
                if len(ArgList) > 0:
                    Metrics.append(FuncList[k](Obj, **ArgList[k]))
                else:
                    Metrics.append(FuncList[k](Obj))
                
            Participants.append(np.hstack(Metrics))
            PID.append(sID)
            bar.update(bar_count)
            bar_count = bar_count + 1
        Results.append(np.squeeze(Participants))
        SubIDs.append(PID)
    progressbar.streams.flush()
    progressbar.streams.wrap_stdout()
    redirect_stdout=True
    bar.finish()
    return Results, SubIDs