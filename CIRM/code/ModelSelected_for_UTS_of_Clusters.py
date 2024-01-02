import pandas as pd
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF,Matern
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.model_selection import train_test_split


def gaussian_model():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF, Matern
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    # parameter = "C(1, (1e-4, 10000)) * RationalQuadratic(alpha=0.01, length_scale_bounds=(1e-5, 20))"
    parameter = ' gaussian_model '
    # kernel_5 = C(1, (1e-4, 1)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.01, 2000))
    n_feats = 8
    kernel = Matern(length_scale=np.array([2]*n_feats),nu=1.5)+WhiteKernel()
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
    return model, parameter

def svr_model():
    from sklearn.svm import SVR
    parameter = ' svr_model '
    model = SVR(kernel='rbf', C=14000, gamma='auto')
    return model, parameter

def random_forest_model():
    #random forest model
    from sklearn.ensemble import RandomForestRegressor
    parameter = ' RandomForest_model '
    # model = RandomForestRegressor(n_estimators=15, max_depth=4, criterion='mae', bootstrap=True)\
    #10 ,6
    model = RandomForestRegressor(n_estimators=15,criterion='squared_error',random_state=2,n_jobs=-1)
    return model, parameter

import pickle
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
for index in range(0, 4):
        df = pd.read_excel("cluster_data-UTS-2.xls", sheet_name="cluster_" + str(index))
        feature_name = [column for column in df][:]
        df_UTS = pd.read_excel("SLM_Ti6Al4V-UTS-2.xlsx")
        scaler = StandardScaler()
        scaler.fit(df_UTS)
        df = scaler.transform(df)
        X = df[:, 1:-1]
        y = df[:, -1]
        if index == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16254)

            train = np.column_stack((X_train, y_train))
            test = np.column_stack((X_test, y_test))
            np.savetxt("X_train-UTS-1.csv", train, delimiter=',',header='Powder_Size,Laser_Spot,Laser_Power,Scanning_Speed,Hatch_Distance,Layer_Thickness,Sample_Area,Sample_Length,Ultimate_Tensile_Strength')
            np.savetxt("X_test-UTS-1.csv", test, delimiter=',',header='Powder_Size,Laser_Spot,Laser_Power,Scanning_Speed,Hatch_Distance,Layer_Thickness,Sample_Area,Sample_Length,Ultimate_Tensile_Strength')

            model, para = svr_model()
            model.fit(X_train,y_train)
            joblib.dump(model, "UTS_cluster_1SVR.model")
        elif index == 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=600)

            train = np.column_stack((X_train, y_train))
            test = np.column_stack((X_test, y_test))
            np.savetxt("X_train-UTS-2.csv", train, delimiter=',',header='Powder_Size,Laser_Spot,Laser_Power,Scanning_Speed,Hatch_Distance,Layer_Thickness,Sample_Area,Sample_Length,Ultimate_Tensile_Strength')
            np.savetxt("X_test-UTS-2.csv", test, delimiter=',',header='Powder_Size,Laser_Spot,Laser_Power,Scanning_Speed,Hatch_Distance,Layer_Thickness,Sample_Area,Sample_Length,Ultimate_Tensile_Strength')

            model, para = random_forest_model()
            model.fit(X_train,y_train)
            joblib.dump(model, "UTS_cluster_2RFR.model")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=253)

            train = np.column_stack((X_train, y_train))
            test = np.column_stack((X_test, y_test))
            np.savetxt("X_train-UTS-3.csv", train, delimiter=',',header='Powder_Size,Laser_Spot,Laser_Power,Scanning_Speed,Hatch_Distance,Layer_Thickness,Sample_Area,Sample_Length,Ultimate_Tensile_Strength')
            np.savetxt("X_test-UTS-3.csv", test, delimiter=',',header='Powder_Size,Laser_Spot,Laser_Power,Scanning_Speed,Hatch_Distance,Layer_Thickness,Sample_Area,Sample_Length,Ultimate_Tensile_Strength')

            model, para = gaussian_model()
            model.fit(X_train, y_train)
            joblib.dump(model, "UTS_cluster_3GPR.model")