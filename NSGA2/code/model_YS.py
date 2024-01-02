import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

def YS(to_predict):
    cluster_centers = [([-0.48846264, -2.30697508, -0.55465694, 0.36368053, 0.55062878, -0.58399231,
                         3.90989886, 2.02018429]),
                       ([-0.07157402, 1.47778281, 0.39201436, 0.66536227, 1.8690567, -0.71297244,
                         0.34930012, -0.53299753]),
                       ([-0.16679138, 0.13063603, 0.76542495, 0.10729315, -0.02530694, 1.15082811,
                         -0.09163938, 0.51330978]),
                       ([0.22516516, -0.31973075, -0.7652522, -0.33253696, -0.5842466, -0.79217076,
                         -0.41942382, -0.52385686])]

    original_mean_X = [35.78651934, 80.24309392, 224.46961326, 926.40331492, 90.62983425,
                       41.85082873, 12.45226519, 23.85773481]
    original_mean_y = 1005.20005209
    original_std_X = [7.49600693, 14.19308521, 110.59919881, 362.19064573, 22.69254008,
                      16.01190379, 13.95310638, 9.97050877]
    original_std_y = 127.40710208262456



    data= (to_predict - original_mean_X) / original_std_X
    cluster_distance = [np.sqrt(np.sum((data-value)**2, axis=1)) for value in cluster_centers]#样本到各个簇之间的距离
    cluster_index = np.argmin(cluster_distance)
    print("归为第几簇：", cluster_index)
    if cluster_index == 0:
        index2 = "0SVR"
    elif cluster_index == 1:
        index2 = "1GPR"
    elif cluster_index == 2:
        index2 = "2RFR"
    else:
        index2 = "3GPR"
    clf = joblib.load("YS_cluster_" + index2+ ".model")
    y_predict = clf.predict(data)*original_std_y+original_mean_y
    return y_predict


