from sklearn.cluster import KMeans,MiniBatchKMeans
import numpy as np
import os


def Kluster(list_good,list_bad):
    os.environ["OMP_NUM_THREADS"] = '1'
    X = list_good[0]
    for i in range(1,len(list_bad)):
        X= np.concatenate((X,list_good[i]))

    for i in range(0,len(list_bad)):
        X= np.concatenate((X,list_bad[i]))
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)

    print("predict____________________")
    good_0 = 0
    good_1 = 0
    bad_0 = 0
    bad_1 = 0
    for i in list_good:
        if kmeans.predict(i)[0] == 0:
            good_0 +=1
        else:
            good_1 += 1
    
    for j in list_bad:
        if kmeans.predict(j)[0] == 0:
            bad_0 +=1
        else:
            bad_1 +=1

    print(f"good | 0 : {good_0} | 1 : {good_1}")
    print(f"bad  | 0 : {bad_0}  | 1 : {bad_1}")