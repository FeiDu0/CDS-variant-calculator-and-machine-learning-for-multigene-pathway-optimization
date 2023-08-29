from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import pandas as pd
import numpy as np
import os
import sys
import random
from iteration_utilities import random_product
from utils.models import svm_train, random_forest_train, mlp_regressor_train, xgboost_train, gpr_train
import yaml


seed = int(sys.argv[1])
config_file = sys.argv[2]
train_data_file = sys.argv[3]
eval_data_file = sys.argv[4] if len(sys.argv) > 4 else None
random.seed(seed)
print("seed:", seed, "\n")

category = 7
# read data from file
train_data = pd.read_csv(train_data_file)
X = train_data.iloc[:, :category]
y = train_data.iloc[:, category]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

if eval_data_file:
    predict_data = pd.read_csv(eval_data_file)
    top_k = len(predict_data)
    repeat = top_k
    sort = False
else:
    with open(config_file, 'r') as file:
        data = yaml.safe_load(file)

    min_value = data['min_value']
    max_values = data['max_values']
    repeat = data['repeat']
    top_k = data['top_k']
    max_copy = data['max_copy']
    ranges = [list(range(min_value, max_value)) for max_value in max_values]
    sort = True
    random_data = random_product(*ranges, repeat=repeat)
    random_data = np.array(list(random_data))
    random_data = random_data.reshape(-1, category)
    random_data = [item for item in random_data if sum(item) <= max_copy]
    predict_data = pd.DataFrame(random_data)
predict_data.columns = X_test.columns
gpr_model = gpr_train(X_train, y_train)
gpr_pred = gpr_model.predict(X_test)

rfr_model = random_forest_train(X_train, y_train)
rfr_pred = rfr_model.predict(X_test)

xgb_model = xgboost_train(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

gpr_model = gpr_train(X_train, y_train)
gpr_pred = gpr_model.predict(X_test)

def predict(model, random_data, top_k, sort):
    result = random_data.copy()
    predict_values = model.predict(result)
    result["Predict_EPA"] = predict_values
    # ensure output dir exists
    if not os.path.exists("output"):
        os.makedirs("output")
    if sort == True:
        result = result.sort_values(by="Predict_EPA", ascending=False)
        if top_k:
            result = result.head(top_k*3)
            result = result.drop_duplicates(subset=["a", "b", "c", "d", "e", "f", "g"])
            result = result.head(top_k)
    result.to_csv(f"output/{type(model).__name__}_{top_k}_result.csv")
    return result
# allow print full dataframe
pd.set_option('display.max_rows', None)
print()
print(f"RFR Predict({top_k}/{repeat}):")
print(predict(rfr_model, predict_data, top_k, sort))
print()
print(f"XGBoost Predict({top_k}/{repeat}):")
print(predict(xgb_model, predict_data, top_k, sort))
print()
print(f"GPR Predict({top_k}/{repeat}):")
print(predict(gpr_model, predict_data, top_k, sort))
print()
