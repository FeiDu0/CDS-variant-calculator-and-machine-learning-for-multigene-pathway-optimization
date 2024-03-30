from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import sys
import random
from utils.models import svm_train, mlp_regressor_train, gpr_train

seed = int(sys.argv[1])
train_data_file = sys.argv[2]
random.seed(seed)
print(f"seed: {seed}")

train_data = pd.read_csv(train_data_file)
category = train_data.shape[1] - 1
print(f"category: {category}")
X = train_data.iloc[:, :category]
y = train_data.iloc[:, category]

svm_mse = []
mlp_mse = []
gpr_mse = []

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    svm_model = svm_train(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_mse.append(mean_squared_error(y_test, svm_pred))

    mlp_model = mlp_regressor_train(X_train, y_train)
    mlp_pred = mlp_model.predict(X_test)
    mlp_mse.append(mean_squared_error(y_test, mlp_pred))

    gpr_model = gpr_train(X_train, y_train)
    gpr_pred = gpr_model.predict(X_test)
    gpr_mse.append(mean_squared_error(y_test, gpr_pred))

print("SVM MSE: \t", round(np.mean(svm_mse), 3))
print("MLP MSE: \t", round(np.mean(mlp_mse), 3))
print("GPR MSE: \t", round(np.mean(gpr_mse), 3))

