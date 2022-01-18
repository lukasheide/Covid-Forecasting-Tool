import pandas as pd
import numpy as np
from sklearn import linear_model, svm, tree
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from Backend.Evaluation.metrics import compute_evaluation_metrics

def create_tuple(infl_factors, y):
    # 1) extend infl_factors with 2 rows, so including beta_t-1 is easier
        #df.loc[-1] = [2, 3, 4]  # adding a row
        #df.index = df.index + 1  # shifting index
        #df = df.sort_index()  # sorting by index
    # 2)insert a new column in a dataframe:
    #infl_factors.insert(2, 'beta_t-1', y_t-1, True)

    # 3) create new list X and y, starting in row 2
    #x = infl_factors.drop([0])

    x = [0, 0]
    i = 1
    while i < len(infl_factors):
        x[i-1] = [infl_factors[i][0], infl_factors[i][1], y[i-1]]
        i += 1
    y_new = [0, 0]
    for j in range(len(y)-1):
        y_new[j] = y[j+1]
    x = pd.DataFrame(x)
    y_new = pd.DataFrame(y_new)

    # 4) split into test and training data
    X_train, X_test, y_train, y_test = train_test_split(x, y_new, random_state=1)
    return X_train, X_test, y_train, y_test

def ml_training(x, y, model):
    if model == "linear_regression":
        reg = linear_model.LinearRegression()
    elif model == "lasso":
        reg = linear_model.Lasso(alpha=0.1)
    elif model == "support_vector_machine":
        reg = svm.SVR()
    elif model == "linear_regression_tree":
        reg = tree.DecisionTreeRegressor()
    elif model == "ensemble_method_adaboost":
        y_array = y.to_numpy()
        y_array = y_array.ravel()
        reg = AdaBoostRegressor(random_state=0, n_estimators=100)
        reg.fit(x, y_array)
        return reg
    elif model == "neural_network":
        y_array = y.to_numpy()
        y_array = y_array.ravel()
        reg = MLPRegressor(random_state=1, max_iter=500).fit(x, y_array)
        return reg
    elif model == "random_forest_classifier":
        reg = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
        y_array = y.to_numpy()
        y_array = y_array.ravel()
        reg = reg.fit(x, y_array)
        return reg
    reg.fit(x, y)
    return reg

def ml_testing(X_test, y_test, model):
    pred = model.predict(X_test)
    scores = compute_evaluation_metrics(y_test,pred)
    return(scores["rmse"])

def predict_beta(model):
    return model.predict([[26, 15, 0.45]])

def run_train_test(X_train, y_train, X_test, y_test, modeltype):
    model = ml_training(X_train, y_train, modeltype)
    return(ml_testing(X_test, y_test, model)), model

def run_all(X_train, y_train, X_test, y_test):
    rmse = []
    min = 1000
    metrics =["linear_regression", "lasso", "support_vector_machine", "linear_regression_tree", "ensemble_method_adaboost", "neural_network", "random_forest_classifier"]
    for i in range(len(metrics)):
        rmse.append(run_train_test(X_train, y_train, X_test, y_test, metrics[i]))
    for i in range(len(rmse)):
        if rmse[i][0] < min:
            min = rmse[i][0]
            best_model = rmse[i][1]
    return best_model

X_train, X_test, y_train, y_test = create_tuple([[25, 12], [26, 14], [27, 15]], [0.47, 0.45, 0.5])
print(len(X_train))
print(len(y_train))
fitted_model = run_all(X_train, y_train, X_test, y_test)
print(predict_beta(fitted_model))