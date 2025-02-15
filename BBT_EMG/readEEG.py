import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.io as sio
from joblib import dump, load

class np_extract_emg:

    def __init__(self, filename, isMEP=True):
        self.filename = filename
        self.file_size = os.path.getsize(filename)
        self.init = 1
        self.fp_data = []
        self.fp_header = []
        self.fp_marker = []
        self.fp_text = []
        self.K_channels = []
        self.N = []
        self.sub_id = []
        self.time_stamp_str = []
        self.loc = []
        self.fid=open(filename,'rb')
        self.process_headers()
        self.process_channels()
        self.get_subject_name()
        self.load_data()

        if isMEP:
            self.process_trigger_data()
            self.get_muscle_names()
            self.parse_meps()
            self.save_meps()
        

    def process_headers(self):

        
        status = self.fid.seek(16)
        self.fp_data = int.from_bytes(self.fid.read(4), byteorder='little')
        self.fp_header = int.from_bytes(self.fid.read(4), byteorder='little')
        self.fp_marker = int.from_bytes(self.fid.read(4), byteorder='little')
        self.fp_text = int.from_bytes(self.fid.read(4), byteorder='little')

        status = self.fid.seek(self.fp_header)

        self.K_channels = int.from_bytes(self.fid.read(2), byteorder='little')
        ignore = self.fid.read(19)
        fp_vector=[self.fp_text, self.fp_marker, self.fp_header, self.fp_data, self.file_size]
        fp_vector.sort()
        idx = fp_vector.index(self.fp_data)
        SIZEOF_FLOAT=4
        self.N = (fp_vector[idx+1] - fp_vector[idx])/SIZEOF_FLOAT/self.K_channels

        status = self.fid.seek(self.fp_header + 39)
        fa_str = self.fid.read(19)
        fa_str = fa_str.decode("utf-8")
        #import pdb; pdb.set_trace()
        self.fa = float(fa_str[2:])
    
    def get_subject_name(self):

        status = self.fid.seek(self.fp_text)
        t = self.fid.readline()
        t = self.fid.readline()
        t = t.decode('utf-8')

        self.sub_id = t.split('=')[1].strip()
        
    def process_channels(self):
        
        self.channels = []
        for i in range(self.K_channels):
            status=self.fid.seek(self.fp_header+35+(i)*203+166)
            h = self.fid.read(8)
            h = h.decode("utf-8")
            hh=h[1:]
            hh = hh.rstrip('\x00')
            #print(hh)
            self.channels.append(hh)
        

    def load_data(self):
        
        #self.data = np.zeros((int(self.N), self.K_channels))
        #self.data= np.float32(self.data)
        self.t = np.linspace(0,self.N,int(self.N))/self.fa
        status= self.fid.seek(self.fp_data+0*self.K_channels*4)
        SIZEOF_FLOAT=4
        self.data = np.fromfile(self.fid, dtype=np.float32, count=self.K_channels*int(self.N))
        self.data = self.data.reshape(int(self.N),self.K_channels)
        self.fid.close()

        
    def process_trigger_data(self):
        
        trig_ch_idx = self.channels.index('DTRIG')
        trig_data = self.data[:,trig_ch_idx]
        diff_sig = np.diff(trig_data)
        stim_idx = np.where(diff_sig==1)
        self.stim_idx = np.squeeze(stim_idx)

    def get_muscle_names(self):

        info_file = os.path.splitext(self.filename)[0] + '.EE_'
        inf_fid = open(info_file,'rt')
        
        #Skip first 2 lines in file
        inf_fid.readline()  
        inf_fid.readline()
        raw_info = inf_fid.readline()
        inf_fid.close()
        parts = raw_info.split(';')
        muscle_info = []
        for i in range(1,len(parts)-1):
            mus_elec = parts[i].split('#')
            elec = mus_elec[1].split('-')
            muscle_name = mus_elec[0]
            e1_name = elec[0]
            e2_name = elec[1]
            muscle_info.append([muscle_name, e1_name, e2_name])
        self.muscle_info = muscle_info
    
    def ms_duration_to_samples(self, t):
        return int(self.fa * t/1000)
    
    def parse_meps(self, pre_pulse = 200, post_pulse= 400):

        self.loc = []
        timing_info = 1
        for m_choice in range(len(self.muscle_info)):
            sel_muscle_name = self.muscle_info[m_choice][0]
            sel_e1 = self.muscle_info[m_choice][1]
            sel_e2 = self.muscle_info[m_choice][2]
            #import pdb; pdb.set_trace()

            ch1_idx =  self.channels.index(sel_e1)
            ch2_idx =  self.channels.index(sel_e2)
            data_sample_len = self.ms_duration_to_samples(400)  #100 ms
            emgData = []
            testpulseframe = []
            for stim_num_idx in range(len(self.stim_idx)):    
                start_idx = self.stim_idx[stim_num_idx] - self.ms_duration_to_samples(pre_pulse)
                stop_idx = self.stim_idx[stim_num_idx] + self.ms_duration_to_samples(post_pulse)
                sig1 = self.data[start_idx:stop_idx, ch1_idx]
                sig2 = self.data[start_idx:stop_idx, ch2_idx]
                sig_time = self.t[start_idx:stop_idx]
                sig_stim_time = self.t[self.stim_idx[stim_num_idx]]
                sig = sig1 - sig2
                #plt.plot(sig - np.mean(sig))
                #plt.show()
                pulseframe = np.ceil((pre_pulse/1000)*self.fa)
                #quantify_mep(sig, int(pulseframe), self.fa)
                testpulseframe.append(pulseframe)

                emgData.append(sig)
            emgData = np.array(emgData)
            testpulseframe = np.array(testpulseframe)
            timing_info = {"testpulseframe": testpulseframe, "freq": self.fa}
            mus_record = {"emgData": emgData, "timing_info": timing_info, "muscle":sel_muscle_name}
            self.loc.append(mus_record)
    
    def save_meps(self):

        self.time_stamp_str = os.path.splitext(filename)[0]
        py_out_file =  self.sub_id + '_' + self.time_stamp_str + '.dat'
        mat_out_file = self.sub_id + '_' + self.time_stamp_str + '.mat'
        dump(self.loc, py_out_file)
        sio.savemat( mat_out_file, {"loc": self.loc})        #In Matlab, need to use: X = [loc{:}]

