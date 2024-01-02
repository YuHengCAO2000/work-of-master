import numpy as np
import pandas as pd
import xlrd


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
    return array_data[:, 1:-3]


def train_cluster(train_data, array_data, source_data):
    from sklearn.cluster import KMeans, DBSCAN
    # model = DBSCAN()
    model = KMeans(n_clusters=3)
    model.fit(train_data)
    label = np.zeros((len(model.labels_), 1), dtype=int)
    for i in range(len(model.labels_)):
        label[i, 0] = int(model.labels_[i])
    # print(train_data, model.cluster_centers_)
    # r = pd.concat([source_data, pd.Series(model.labels_, index=source_data.index)], axis=1)

    # print(labels)
    combine = np.concatenate((array_data, label), axis=1)
    writer = pd.ExcelWriter('../file/cluster_dataset.xls')
    r0 = pd.concat([pd.DataFrame(array_data[:, 0:12]), pd.DataFrame(model.labels_)], axis=1)
    r0.columns = ['Alloy', 'Powder_Size', 'Laser_Spot','Laser_Power','Scanning_Speed','Hatch_Distance','Layer_Thickness','Sample_Area','Sample_Length','Ultimate_Tensile_Strength','Yield_Strength','Elongation','label']
    r0.to_excel(writer, sheet_name='cluster_label')
    for i in range(len(np.unique(model.labels_))):
        cluster_subset = combine[combine[:, -1] == i][:, :-1]
        # print(np.arange(0, len(cluster_subset[:, 0])+1, 1).T)
        r0 = pd.DataFrame(np.arange(0, int(len(cluster_subset[:, 0])), 1).T)
        r1 = pd.DataFrame(cluster_subset)
        r = pd.concat([r0, r1], axis=1)
        r.columns = ['Alloy'] + list(source_data.columns)
        r.to_excel(writer, sheet_name='cluster_'+str(i))
    plot_cluster(train_data, model.labels_)
    writer.save()


def plot_cluster(data_zs, r):
    from sklearn.manifold import TSNE

    tsne = TSNE()
    tsne.fit_transform(data_zs)  # 进行数据降维,降成两维
    # a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
    tsne = pd.DataFrame(tsne.embedding_)  # 转换数据格式

    import matplotlib.pyplot as plt

    d = tsne[r == 0]
    plt.plot(d[0], d[1], 'r.',marker = '^',alpha = 0.7,markersize = 10)


    d = tsne[r == 1]
    plt.plot(d[0], d[1], 'b.',marker = 'd',alpha = 0.7,markersize = 10)


    d = tsne[r == 2]
    plt.plot(d[0], d[1], 'g.',marker = 'o',alpha = 0.7,markersize = 10)
    plt.show()

def run_cluster():
    # print("run_cluster")
    original_data = pd.read_excel('SLM_Ti6Al4V-2.xlsx')
    original_data = original_data.values
    resource_data, np_data = load_data('../file/SLM_Ti6Al4V-2.xlsx')
    array_data = data_process(np_data,original_data)
    train_data = cluster_split(array_data)
    # print(array_data[1, :])
    # print(train_data[1,:])
    # print(array_data, np_data)
    train_cluster(train_data, np_data, resource_data)


if __name__ == "__main__":
    print('welcome to cluster world')
    run_cluster()






