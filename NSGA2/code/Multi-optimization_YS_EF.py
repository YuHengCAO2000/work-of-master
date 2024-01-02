import model_YS
import model_EF
import optuna
import numpy as np
import joblib

def objective(trial):
    #优化参数
    powder_size = trial.suggest_float("powder_size", 37.7, 37.7)
    laser_spot = trial.suggest_int("laser_spot", 75, 75)
    laser_power = trial.suggest_int("laser_power", 100, 300)
    scanning_speed = trial.suggest_int("scanning_speed", 800, 2000)
    hatch_distance = trial.suggest_int("hatch_distance", 30, 150)
    layer_thickness = trial.suggest_int("layer_thickness", 30, 30)
    sample_area = trial.suggest_float("sample_area", 2, 2)
    standard_distance = trial.suggest_int("sample_length", 8, 8)

    #估算参数相应值
    to_predict = np.array([powder_size,laser_spot,laser_power,scanning_speed,hatch_distance,layer_thickness,sample_area,standard_distance]).reshape((-1,8))
    YS_predict = model_YS.YS(to_predict).reshape(-1, 1)
    El_predict = model_EF.El(to_predict).reshape(-1, 1)
    #UTS = UTS_predict
    YS = YS_predict
    Elongation = El_predict
    return Elongation,YS

algorithm = optuna.samplers.NSGAIISampler()

create_study = optuna.create_study(
    directions=["maximize","maximize"],
    sampler=algorithm,)

create_study.optimize(objective, n_trials=10000, timeout=600)
create_study = joblib.dump(create_study, "../file/study_NSGAII.pkl")
create_study = joblib.load("../file/study_NSGAII.pkl")

print("Number of finished trials:", len(create_study.trials))
print("Pareto front:")
trials = sorted(create_study.best_trials, key=lambda t: t.values)
for trial in trials:
    print("  Trial#{}".format(trial.number))
    print("    Params: {}".format(trial.params))

plt = optuna.visualization.plot_pareto_front(create_study, target_names=["Elongation to Fracture","Yield Strength"])
plt.show()