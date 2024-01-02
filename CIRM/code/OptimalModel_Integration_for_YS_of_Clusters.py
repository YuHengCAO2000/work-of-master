import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

df_original = pd.read_excel("SLM_Ti6Al4V-YS-1.xlsx")
original_data = df_original.values[:, 1: ]
print(original_data)
original_mean = np.mean(original_data, axis=0)
print("original_mean", original_mean)
original_mean_X = original_mean[:-1]
print("original_mean_X", original_mean_X)
original_mean_y = original_mean[-1]
print("original_mean_y", original_mean_y)
original_std = np.std(original_data, axis=0)
original_std_X = original_std[:-1]
print("original_std_X", original_std_X)
original_std_y = original_std[-1]
print("original_std_y", original_std_y)
original_min = np.min(original_data, axis=0)
original_max = np.max(original_data, axis=0)
#print("11111", original_max)
#print("22222", original_min)
#print(original_min.shape)
#print(original_max-original_min)

cluster_centers = []
for index in range(3):
    #每个簇上的样本
    cluster_data = pd.read_excel("cluster_data-YS-2.xls",
                                 sheet_name="cluster_"+str(index))
    #print("222ddddd", cluster_data.values)
    cluster_dataX = cluster_data.values[:, 1:9]
    normalize_data = [(row-original_mean_X)/original_std_X
                      for row in cluster_dataX]#每个簇按照全部标准归一化
    #求每个簇的聚类中心
    #print("每个簇标准化的数据为", normalize_data)
    cluster_center = np.mean(normalize_data, axis=0)
    cluster_centers.append(cluster_center)

cluster_centers = np.array(cluster_centers)#将聚类中心转化为numpy数组
print("聚类中心为", cluster_centers)
df1 = pd.read_excel("train_YS.xlsx")
df = pd.read_excel("test_YS.xlsx")
validation_data1 = df1.values
Y1 = validation_data1[:, -1]
validation_dataX1, validation_targetY1 = validation_data1[:, :-1], validation_data1[:, -1]
validation_data = df.values
Y = validation_data[:, -1]
validation_dataX, validation_targetY = validation_data[:, :-1], validation_data[:, -1]
# normalize_data, normalize_target = pre_progressing(validation_data)
#第一条样本
#print("测试集X",validation_dataX)
#print("测试集Y",validation_targetY)
#训练集R^2的确认
target1_YS = []
predict1 = []
predicts1 = []
for i in range(0, 137):
    data1, target1 = (validation_dataX1[i:i+1]-original_mean_X)/original_std_X, (validation_targetY1[i:i+1]-original_mean_y)/original_std_y
    cluster_distance = [np.sqrt(np.sum((data1-value)**2, axis=1))
                        for value in cluster_centers]#样本到各个簇之间的距离
    print("样本到各个簇上的距离", cluster_distance)
    cluster_dis = [dis[0] for dis in cluster_distance]
    print(cluster_dis)
    cluster_index = np.argmin(cluster_distance)
    print("该样本所属簇的标号为", cluster_index)#从0开始计算
    if cluster_index == 0:
        index2 = "1SVR"
    elif cluster_index == 1:
        index2 = "2RFR"
    else:
        index2 = "3GPR"
    clf = joblib.load("YS_cluster_" + index2+ ".model")
    print("true value", target1*original_std_y+original_mean_y)
    print(target1*original_std_y+original_mean_y, clf.predict(data1)*original_std_y+original_mean_y)
    target1_YS.append(target1)
    predict1.append(clf.predict(data1))
    predicts1.append(clf.predict(data1)*original_std_y+original_mean_y)
predicts1 = pd.DataFrame(predicts1)
predicts1.to_excel("predicts1_YS.xlsx")
#放入数据进行分类与预测
target_YS = []
predict = []
predicts = []
print("test，test")
for i in range(0, 36):
    data, target = (validation_dataX[i:i + 1] - original_mean_X) / original_std_X, (
                validation_targetY[i:i + 1] - original_mean_y) / original_std_y
    cluster_distance = [np.sqrt(np.sum((abs(data-value))**2, axis=1))
                        for value in cluster_centers]#样本到各个簇之间的距离
    print("样本到各个簇上的距离", cluster_distance)
    cluster_dis = [dis[0] for dis in cluster_distance]
    print(cluster_dis)
    cluster_index = np.argmin(cluster_distance)
    print("该样本所属簇的标号为", cluster_index)#从0开始计算
    if cluster_index == 0:
        index2 = "1SVR"
    elif cluster_index == 1:
        index2 = "2RFR"
    else:
        index2 = "3GPR"
    clf = joblib.load("YS_cluster_" + index2+ ".model")
    print("true value", target*original_std_y+original_mean_y)
    print(target*original_std_y+original_mean_y, clf.predict(data)*original_std_y+original_mean_y)
    target_YS.append(target)
    predict.append(clf.predict(data))
    predicts.append(clf.predict(data)*original_std_y+original_mean_y)
predicts = pd.DataFrame(predicts)
predicts.to_excel("predicts_YS.xlsx")

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(Y1, predicts1),
        mean_squared_error(Y, predicts)))
print('MAE train: %.3f, test: %.3f' % (
        mean_absolute_error(Y1, predicts1),
        mean_absolute_error(Y, predicts)))
print('分簇多算法自适应回归的R^2 train: %.3f, test: %.3f' % (
        r2_score(Y1, predicts1),
        r2_score(Y, predicts)))
predicts.to_excel("predicts_YS.xlsx")

print('MAE test: %.3f' % (
        mean_absolute_error(Y, predicts)))
print('分簇多算法自适应回归的R^2 train: %.3f, test: %.3f' % (
        r2_score(Y1, predicts1),
        r2_score(Y, predicts)))
