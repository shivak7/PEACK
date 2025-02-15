import numpy as np
from dicttoxml import dicttoxml
import xmltodict
from xml.dom.minidom import parseString

class Dict2Class(object):

    def __init__(self, my_dict):

        for key in my_dict:
            setattr(self, key, my_dict[key])

class Parameter:

    def __init__(self):

        self.DataPath = ""              # Main Data Path
        self.MapFile = ""               # Joint(Body part) map filename
        self.JointNames = []           # Names of joints to consider for analysis
        self.Fs = -1                    # Timeseries(TS)  sampling rate (FPS if video)
        self.TaskSegmentFlag = False    # Whether to segment by task or consider entire TS
        self.TimeSegFile = ""               # Filename for segmenting data by task times
        self.Trunc = []                 # Truncation (pair) range if TaskSegmentFlag is False
        self.OutFile = ""               # Extracted data saved as a Pickle file
        #Filter and TS params

        self.do_filtering = False
        self.do_smoothing = False
        self.smoothing_alpha = 0.1
        self.lowpass_cutoff = 0
        self.lowpass_order = 5
        self.median_filter_win = 0.1
        self.unit_rescale = 1.0
        self.method = 'PEACK'
        self.drop_contiguous_columns = None
        self.interp_missing = True
        self.Use2D = False


def AHAParamsDemo():

    P = Parameter()
    P.DataPath = "C:\\Users\\shiva\\Dropbox\\Burke Work\\DeepMarker\\Processed Data\\AHA2D\\data0\\Body"
    P.MapFile = "OP_BodyMap.csv"
    P.JointNames = ["RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "Neck", "MidHip"]
    P.Fs = 30
    P.TaskSegmentFlag = True
    P.TimeSegFile = "TimeData_I.csv"
    P.Trunc = [1,2]
    P.OutFile = "AHA_2015_Initial_Body.pkl"

    P.do_filtering = False
    P.do_smoothing = False
    P.smoothing_alpha = 0.1
    P.lowpass_cutoff = 3
    P.lowpass_order = 5
    P.median_filter_win = 0.1
    P.unit_rescale = 1.0
    P.method = 'PEACK'
    P.drop_contiguous_columns = -30
    P.interp_missing = True
    P.Use2D = False
    return P

def SaveParameters(Param, fn):

    P = vars(Param)
    xml = dicttoxml(P, attr_type=False, custom_root='Param') # set root node to Person
    dom = parseString(xml)
    #print(dom.toprettyxml())

    xml_file = open(fn, "w")
    xml_file.write(dom.toprettyxml())
    xml_file.close()


def LoadParameters(fn):

    with open(fn) as fd:
        doc = xmltodict.parse(fd.read())

    P = Dict2Class(doc["Param"])
    P.__class__ = Parameter

    #Fix structure post-conversion
    #import pdb; pdb.set_trace()
    P.JointNames = P.JointNames['item']
    P.Fs = float(P.Fs)
    P.Trunc = [eval(i) for i in P.Trunc['item']]
    P.do_filtering = P.do_filtering == "True"
    P.do_smoothing = P.do_smoothing == "True"
    P.smoothing_alpha = float(P.smoothing_alpha)
    P.lowpass_cutoff = int(P.lowpass_cutoff)
    P.lowpass_order = int(P.lowpass_order)
    P.median_filter_win = float(P.median_filter_win)
    P.unit_rescale = float(P.unit_rescale)
    P.drop_contiguous_columns = int(P.drop_contiguous_columns)
   
    if not hasattr(P,'time_unit'):
        P.time_unit = 's'          #Default

    if not hasattr(P, 'interp_missing'):
        P.interp_missing = True         #Default
    else:
        P.interp_missing = P.interp_missing == "True"

    if not hasattr(P,'zeros_as_nan'):
        P.zeros_as_nan = False          #Default
    else:
        P.zeros_as_nan = P.zeros_as_nan == "True"


    P.Use2D = P.Use2D == "True"

    return P
