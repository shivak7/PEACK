from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from dtaidistance import dtw
from sklearn.cluster import HDBSCAN
from matplotlib import pyplot as plt
import matplotlib.cm as cm


class trajectory_segments:

    def __init__(self, ts_data, segment_indices):               ### Prepare trajectory data and separate by dimensionality N-D trajectories (typically 2D or 3D)

        self.Ndims = ts_data.shape[1]
        self.data_dims = []      #  Use a list to store same number of Distance matrices as the number of data dimensions (X, Y, Z etc.)
        self.segment_indices = segment_indices
        self.Nsegments = len(self.segment_indices) - 1
        self.clustering_complete = False
        for n in range(self.Ndims):
            data_segments = []
            for i in range(self.Nsegments):
                segment = ts_data[segment_indices[i]:segment_indices[i+1],n]
                data_segments.append(segment)
            self.data_dims.append(data_segments)
        self.hdb_using_dtw_distance_matrix()

    def hdb_using_dtw_distance_matrix(self):

        ### Compute distance matrix for each dimension

        #distMatrices = [] 
        
        distMatrix = np.zeros((self.Nsegments,self.Nsegments))
        for n in range(self.Ndims):
            dM = dtw.distance_matrix_fast(self.data_dims[n])
            distMatrix = distMatrix + dM**2

        distMatrix = np.sqrt(distMatrix)
        
        hdb = HDBSCAN(metric='precomputed')
        hdb.fit(distMatrix)
        self.cluster_labels = hdb.labels_
        self.clustering_complete = True
        #self.Nclusts = len(np.unique(self.cluster_labels))
        ulabels, self.cluster_count = np.unique(self.cluster_labels, return_counts=True)
        self.Nclusts = len(ulabels)
        #print(self.Nclusts , 'clusters found.')

    def show_clusters(self, colormap_name=None):

        fig = plt.figure()
        if self.Ndims == 3:
            ax = fig.add_subplot(111, projection='3d')
        elif self.Ndims == 2:
            ax = fig.add_subplot(111)
        else:
            print('Data dimensions are', self.Ndims, '. Cannot plot as trajectories.')
            raise IndexError
        
        if colormap_name is None:
            colormap_name = 'viridis'

        cmap = cm.get_cmap(colormap_name)
        #colors = [cmap(i) for i in np.linspace(0, 1, self.Nclusts)]
        colors = ['b','r', 'k', 'g', 'c', 'm', 'y', 'w', 'w', 'w', 'w', 'w', 'w']
        for i in range(self.Nsegments):
            
            X = self.data_dims[0][i]
            Y = self.data_dims[1][i]
            
            if self.Ndims == 3:
    
                Z = self.data_dims[2][i]
                ax.plot(X, Y, Z, colors[self.cluster_labels[i]])
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')


            elif self.Nidms == 2:
                ax.plot(X, Y, colors[self.cluster_labels[i]])

        plt.show()


        




