from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def svm_train(X_train, y_train):
    clf = svm.SVR(kernel='rbf', C=1, gamma='scale')
    clf.fit(X_train, y_train)
    return clf


def mlp_regressor_train(X_train, y_train):
    mlp = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=1000)
    mlp.fit(X_train, y_train)
    return mlp


def gpr_train(X_train, y_train):
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train, y_train)
    return gpr

