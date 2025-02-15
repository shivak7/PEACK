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

class SubDirectories:

    def __init(self, n):
        a = 5

class DataLoader:

    def __init__(self, MainPath):

        self.MainPath = MainPath

        lst = [f for f in os.listdir(self.MainPath) if os.path.isdir(os.path.join(self.MainPath, f))]
        self.SubDirs = []
        self.NsubDirs = len(lst)
        lst = sorted(lst)

        if len(lst)==0:
            self.NsubDirs = 1
            lst = '.' + os.sep
            subd = SubDirectories()
            subd.name = lst
            FullInPath = os.path.join( self.MainPath, subd.name)
            subd.Files = [f for f in os.listdir(FullInPath) if os.path.isfile(os.path.join(FullInPath, f))]
            #subd.Files = sorted(subd.Files)
            subd.Files.sort(key=natural_keys)
            subd.Nfiles = len(subd.Files)
            self.SubDirs.append(subd)

        else:
            for i in range(len(lst)):

                subd = SubDirectories()
                subd.name = lst[i]
                FullInPath = os.path.join( self.MainPath, subd.name)
                
                subd.Files = [f for f in os.listdir(FullInPath) if os.path.isfile(os.path.join(FullInPath, f))]
                if '.DS_Store' in subd.Files:
                    idx = subd.Files.index('.DS_Store')
                    del subd.Files[idx]
                #import pdb; pdb.set_trace()
                #subd.Files = sorted(subd.Files)
                subd.Files.sort(key=natural_keys)
                subd.Nfiles = len(subd.Files)
                self.SubDirs.append(subd)

    def getFile(self, subdir_idx, file_idx):

        fn = os.path.join(self.MainPath, self.SubDirs[subdir_idx].name, self.SubDirs[subdir_idx].Files[file_idx])
        return fn
