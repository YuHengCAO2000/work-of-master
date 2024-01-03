# work-of-master
This is a project: prediction for mechanical and optimization for process parameters in L-PBF
## English:
### Program description:
#### CIRM Part:

(1) Silhouette_Coefficient.py: Determining the optimal number of clusters required for the K-Means clustering method;

(2) Cluster_Ti6Al4V.py: Clusters and divides the collected and organized data of 173 groups of Ti6Al4V alloy;

(3) ModelSelected_for_EF(UTS/YS)_of_Clusters.py: Builds and selects the optimal prediction models for UTS/YS/EF in each cluster;

(4) OptimalModel_Integration_for_EF(UTS/YS)_of_Clusters.py: Integrates the optimal prediction models for each cluster;

(5) SHAP for EF(UTS/YS): SHAP analysis on the integrated prediction models for EF(UTS/YS).

#### NSGA2 Part:

(1) model_EF(YS).py: Subfile used to connect the CIRM model with the NSGA2 model;

(2) ModelSelected_for_EF(UTS/YS)_of_Clusters.py: multi-objective optimization based on the integrated optimal prediction models for EF and YS obtained from CIRM.

#### !!! Python program installation and configuration of relevant libraries are required in advance.



## Chinese：
### 程序说明：
#### CIRM部分：
（1）Silhouette_Coefficient.py：确定K-Means聚类方法所需要聚类的最佳数目；

（2）Cluster_Ti6Al4V.py：将收集整理好的173组Ti6Al4V合金数据进行聚类划分；

（3）ModelSelected_for_EF(UTS/YS)_of_Clusters.py：构建与选择各聚类中UTS/YS/EF的最优预测模型；

（4）OptimalModel_Integration_for_EF(UTS/YS)_of_Clusters.py：各聚类最优预测模型的整合；

（5）SHAP for EF(UTS/YS)：对已整合后EF(UTS/YS)的预测模型进行SHAP分析。

#### NSGA2部分：

（1）model_EF(YS).py：用于连接CIRM模型与NSGA2模型的子文件；

（2）ModelSelected_for_EF(UTS/YS)_of_Clusters.py：基于CIRM所得到的EF与YS的整合后的最优预测模型进行多目标优化。

#### ！！！需提前安装python程序并配置好相应的库
