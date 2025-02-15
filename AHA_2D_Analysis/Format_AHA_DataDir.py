import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def check_make_dir(dir_pathname):

    if not os.path.exists(dir_pathname):
        os.makedirs(dir_pathname)


mainPath = '/Users/shiva/Dropbox/Burke Work/DeepMarker/Processed Data/AHA2D/AHAKinematics/AHA_2018/Post'
check_make_dir(mainPath + '/' + "Body")
check_make_dir(mainPath + '/' + "LHand")
check_make_dir(mainPath + '/' + "RHand")

List = os.listdir(mainPath)

for fi in range(len(List)):
    fullFile = mainPath + '/' + List[fi]
    if os.path.isfile(fullFile) == True:
        if (len(os.path.splitext(fullFile))==2) & (os.path.splitext(fullFile)[1].lower()=='.csv'):       #If ext == '.csv' (To be safe, make .ext case insensitive)

            if "body" in List[fi].lower():
                os.rename(fullFile, mainPath + '/' + "Body" + '/' + List[fi])
            elif "left_hand" in List[fi].lower():
                os.rename(fullFile, mainPath + '/' + "LHand" + '/' + List[fi])
            elif "right_hand" in List[fi].lower():
                os.rename(fullFile, mainPath + '/' + "RHand" + '/' + List[fi])
