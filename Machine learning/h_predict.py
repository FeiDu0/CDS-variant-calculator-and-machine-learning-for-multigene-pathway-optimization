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
from utils.models import svm_train, mlp_regressor_train, gpr_train
import yaml


seed = int(sys.argv[1])
config_file = sys.argv[2]
train_data_file = sys.argv[3]
max_copy = sys.argv[4] if len(sys.argv) > 4 else None
test_size = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
eval_data_file = sys.argv[6] if len(sys.argv) > 6 else None
random.seed(seed)
print(f"Seed: {seed}")
print(f"Test Size: {test_size}")

# read data from file
train_data = pd.read_csv(train_data_file)
category = train_data.shape[1] - 1
headers = list(train_data.columns.values[:category])
target = train_data.columns.values[-1]
X = train_data.iloc[:, :category]
y = train_data.iloc[:, category]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

if eval_data_file:
    predict_data = pd.read_csv(eval_data_file)
    top_k = len(predict_data)
    repeat = top_k
    sort = False
else:
    with open(config_file, 'r') as file:
        data = yaml.safe_load(file)

    min_values = data['min_values']
    max_values = data['max_values']
    repeat = data['repeat']
    top_k = data['top_k']
    max_copy = int(max_copy) if max_copy else data['max_copy']
    ranges = [list(range(min_values[i], max_value)) for i, max_value in enumerate(max_values)]
    sort = True
    random_data = random_product(*ranges, repeat=repeat)
    random_data = np.array(list(random_data))
    random_data = random_data.reshape(-1, category)
    random_data = [item for item in random_data if sum(item) <= max_copy]
    predict_data = pd.DataFrame(random_data)
predict_data.columns = X_test.columns
gpr_model = gpr_train(X_train, y_train)
gpr_pred = gpr_model.predict(X_test)

gpr_model = gpr_train(X_train, y_train)
gpr_pred = gpr_model.predict(X_test)

def predict(model, random_data, top_k, sort):
    predict_header = f"Predict_{target}"
    expand_num = 10
    result = random_data.copy()
    predict_values = model.predict(result)
    result[predict_header] = predict_values
    average = 0
    std = 0
    # ensure output dir exists
    if not os.path.exists("output"):
        os.makedirs("output")
    if sort:
        result = result.sort_values(by=predict_header, ascending=False)
        # calculate average and std of predict_header
        average = result[predict_header].mean()
        std = result[predict_header].std()
        if top_k:
            temp_result = pd.DataFrame(columns=result.columns)
            while len(temp_result) < top_k:
                current_top = result.head(top_k * expand_num)
                unique_top = current_top.drop_duplicates(subset=headers)
                temp_result = pd.concat([temp_result, unique_top]).drop_duplicates(subset=headers)
                result = result.iloc[top_k * expand_num:]
                if result.empty:
                    break
            result = temp_result.head(top_k)
    result.to_csv(f"output/{type(model).__name__}_{top_k}_result.csv")
    return result, f"Average: {average}\nStd: {std}"
# allow print full dataframe
pd.set_option('display.max_rows', None)
print()

print(f"GPR Predict({top_k}/{repeat}):")
result, extra = predict(gpr_model, predict_data, top_k, sort)
print(result)
print(extra)
print()
