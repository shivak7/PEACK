def avg_traj(trials, max_length):

    Tlist = []
    # for i in range(len(trials)):
    #
    #     trial = trials[i] #- np.mean(trials[i], axis=0)
    #     trial = stats.zscore(trial, axis=0)
    #     if(len(trial)<max_length):
    #         n_trial = np.zeros((max_length,trial.shape[1]))
    #         for j in range(trial.shape[1]):
    #             n_trial[:,j] = np.interp( np.linspace(0.0, 1.0, max_length, endpoint=False), np.linspace(0.0, 1.0, len(trial), endpoint=False),trial[:,j])
    #     else:
    #         n_trial = trial
    #     #n_trial = np.reshape(n_trial,(1,n_trial.shape[0]*n_trial.shape[1]))
    #     n_trial = np.concatenate((n_trial[:,0],n_trial[:,1], n_trial[:,2]))
    #     Tlist.append(n_trial);
    #
    # Tlist = np.squeeze(np.array(Tlist))

    Nclust = 3;
    #kmeans = KMeans(n_clusters=Nclust, random_state=0).fit(Tlist)

    ts_data = to_time_series_dataset(trials)
    ts_data = stats.zscore(ts_data,axis=1, nan_policy='omit')
    #ts_data = ts_data - np.nanmean(ts_data, axis=1,keepdims=True)
    import pdb; pdb.set_trace()
    #kmeans = TimeSeriesKMeans(n_clusters=Nclust, metric='dtw', random_state=0).fit(ts_data)
    #labels = kmeans.labels_
    gak_km = KernelKMeans(n_clusters=Nclust,
        kernel="gak",
        kernel_params={"sigma": "auto"},
        n_init=20,
        verbose=True,
        random_state=0)
    labels = gak_km.fit_predict(ts_data)


    for i in range(Nclust):

        X = ts_data[labels==i,:,0]
        plt.subplot(Nclust,1,i+1)
        plt.plot(X.T)

    plt.show()

    return 0
    #import pdb; pdb.set_trace();
