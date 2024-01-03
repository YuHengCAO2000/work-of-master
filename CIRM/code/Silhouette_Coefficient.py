import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from sklearn import metrics
import xlrd
import matplotlib.pyplot as plt


def load_data(input_file):
    source_data = pd.read_excel(input_file)
    np_data = source_data.iloc[:,:].values
    # print(source_data.columns)
    return source_data, np_data


def data_process(np_data,original_data):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    scaler.fit(original_data)
    array_data = scaler.transform(np_data)
    return array_data


def cluster_split(array_data):
    # print(array_data[1, 1:-1])
    return array_data[:, 1:-1]


def train_cluster(train_data, array_data, source_data):
    from sklearn.cluster import KMeans, DBSCAN
    # model = DBSCAN()
    model = KMeans(n_clusters=4)
    model.fit(train_data)
    label = np.zeros((len(model.labels_), 1), dtype=int)
    for i in range(len(model.labels_)):
        label[i, 0] = int(model.labels_[i])
    # print(train_data, model.cluster_centers_)
    # r = pd.concat([source_data, pd.Series(model.labels_, index=source_data.index)], axis=1)

    # print(labels)
    combine = np.concatenate((array_data, label), axis=1)
    writer = pd.ExcelWriter('cluster_data.xls')
    r0 = pd.concat([pd.DataFrame(array_data[:, 0:10]), pd.DataFrame(model.labels_)], axis=1)
    r0.columns = ['Alloy', 'Powder_Size', 'Laser_Spot','Laser_Power','Scanning_Speed','Hatch_Distance','Layer_Thickness','Sample_Area','Sample_Length','Ultimate_Tensile_Strength','Yield_Strength','Elongation','label']
    r0.to_excel(writer, sheet_name='cluster_label')
    for i in range(len(np.unique(model.labels_))):
        cluster_subset = combine[combine[:, -3] == i][:, :-3]
        # print(np.arange(0, len(cluster_subset[:, 0])+1, 1).T)
        r0 = pd.DataFrame(np.arange(0, int(len(cluster_subset[:, 0])), 3).T)
        r1 = pd.DataFrame(cluster_subset)
        r = pd.concat([r0, r1], axis=1)
        r.columns = ['Alloy'] + list(source_data.columns)
        r.to_excel(writer, sheet_name='cluster_'+str(i))
    writer.save()



def run_cluster():
    # print("run_cluster")
    original_data = pd.read_excel('SLM_Ti6Al4V-UTS-2.xlsx')
    original_data = original_data.values
    resource_data, np_data = load_data('SLM_Ti6Al4V-UTS-2.xlsx')
    array_data = data_process(np_data,original_data)
    train_data = cluster_split(array_data)
    # print(array_data[1, :])
    # print(train_data[1,:])
    # print(array_data, np_data)
    train_cluster(train_data, np_data, resource_data)
    scoreset = []
    for i in range(2, 9):
        n_cluster = i+6
        estimator = KMeans(n_clusters=n_cluster, random_state=200, max_iter=500)  # 构造聚类器
        estimator.fit(train_data)  # 聚类
        label_pred = estimator.labels_
        score = metrics.silhouette_samples(train_data, label_pred, metric='euclidean')
        result = np.mean(score)
        scoreset.append(result)
        print('i = {}, score = {}'.format(i, result))


if __name__ == "__main__":
    print('welcome to cluster world')
    run_cluster()
