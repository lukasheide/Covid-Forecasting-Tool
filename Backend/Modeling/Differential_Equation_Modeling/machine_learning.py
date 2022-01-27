import math

import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model, svm, tree
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from Backend.Evaluation.metrics import compute_evaluation_metrics

# calculate the mean temperatur for each week to fill the empty rows
def claculate_wind_mean(dataframe):
    dataframe.groupby(['week'])
    week = dataframe['week']
    mean = []
    for j in range(week.max()):
        count = 0
        sum_wind = 0
        for i in range(len(dataframe.index)):
            if math.isnan(dataframe['wind'][i]) == False:
                if dataframe['week'][i] == j:
                    sum_wind = sum_wind + dataframe['wind'][i]
                    count += 1
        if count == 0:
            mean.append(0)
        else:
            mean.append(sum_wind/count)
    print('wind_mean_finished')
    return mean

# calculate the mean temperatur for each week to fill the empty rows
def claculate_temperature_mean(dataframe):
    dataframe.groupby(['week'])
    week = dataframe['week']
    mean = []
    for j in range(week.max()):
        count = 0
        sum_temperature = 0
        for i in range(len(dataframe.index)):
            if math.isnan(dataframe['temperature'][i]) == False:
                if dataframe['week'][i] == j:
                    sum_temperature = sum_temperature + dataframe['temperature'][i]
                    count += 1
        if count == 0:
            mean.append(0)
        else:
            mean.append(sum_temperature/count)
    print('temperature_mean_finished')
    return mean

# fill the empty rows with mean, deleting rows with infections=0
def fill_empty_rows(dataframe):
    #fill empty wind rows
    mean = claculate_wind_mean(dataframe)
    week = dataframe['week']
    for i in range(len(dataframe.index)):
        if math.isnan(dataframe['wind'][i]) == True:
            week = dataframe['week'][i]
            dataframe.at[[i], 'wind'] = mean[week-1]
 #           for j in range(week.max()):
 #               if dataframe['week'][i] == j:
 #                   dataframe.at[[i], 'wind'] = mean[j-1]
 #                   break
    print('wind_rows_finished')

    #fill empty temeprature rows
    mean = claculate_temperature_mean(dataframe)
    week = dataframe['week']
    for i in range(len(dataframe.index)):
        if math.isnan(dataframe['temperature'][i]) == True:
            week = dataframe['week'][i]
            dataframe.at[[i], 'temperature'] = mean[week-1]
 #           for j in range(week.max()+1):
 #               if dataframe['week'][i] == j:
 #                   dataframe.at[[i], 'temperature'] = mean[j-1]
 #                   break
    print('temperature_rows_finished')
    return dataframe

def delete_zero_infections(dataframe):
    dataframe.drop(dataframe[dataframe['infections'] == 0.0].index, inplace=True)
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe

def delete_small_beta(dataframe):
    dataframe.drop(dataframe[dataframe['beta'] <= 0.025].index, inplace=True)
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe

def create_tuple():
    # 1) separate beta, delete districts, transform variant
    df = pd.read_csv("all_matrix_data.csv")
    #separate beta
    beta = df['beta']
    #transform variant
    df_1 = pd.get_dummies(df['variant'])
    df_2 = df.drop(columns='variant')
    df = pd.concat([df_1, df_2], axis=1)
    #delete districts
    df = df.drop(columns='district')

    #delte infections = 0, fill mean in empty temperature/wind cells
    df = delete_zero_infections(df)
    fill_empty_rows(df)

    #dataframe for labels
    beta = df['beta']

    #delete beta
    df = df.drop(columns='beta')

    #delete columns week and unnamed
    df = df.drop(columns='week')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # 4) split into test and training data
    X_train, X_test, y_train, y_test = train_test_split(df, beta, shuffle=True, train_size=0.75, random_state=1)
    df.to_csv("df.csv")
    beta.to_csv("beta.csv")
    X_test.to_csv("x_test.csv")
    X_train.to_csv("x_train.csv")
    y_test.to_csv("y_test.csv")
    y_train.to_csv("y_train.csv")
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
    elif model == "random_forest_regressor":
        reg = RandomForestRegressor(n_estimators=10, random_state=0, max_depth=2)
        y_array = y.to_numpy()
        y_array = y_array.ravel()
        reg = reg.fit(x, y_array)
        return reg
    reg.fit(x, y)
    return reg

#test the fitted model
def ml_testing(X_test, y_test, model):
    pred = model.predict(X_test)
    scores = compute_evaluation_metrics(y_test,pred)
    return(scores["rmse"])

#test the fitted model: coompare pred and beta t-1
def ml_testing_beta(X_test, y_test):
    beta_t_1 = X_test['beta_t_minus_1']
    pred = model.predict(X_test)
    scores = compute_evaluation_metrics(y_val=y_test,y_pred=pred)
    scores_beta = compute_evaluation_metrics(y_val=y_test, y_pred=beta_t_1)
    return(scores["rmse"], scores_beta["rmse"])

#predict with given model
def predict_beta(model, predictions):
    return model.predict([predictions])

#first train, then run the model
def run_train_test(X_train, y_train, X_test, y_test, modeltype):
    model = ml_training(X_train, y_train, modeltype)
    return(ml_testing(X_test, y_test, model)), model

#run all model types to find best model
def run_all(X_train, y_train, X_test, y_test):
    rmse = []
    min = 100000
    metrics =["linear_regression", "lasso", "support_vector_machine", "linear_regression_tree", "ensemble_method_adaboost", "neural_network", "random_forest_regressor"]
    for i in range(len(metrics)):
        rmse.append(run_train_test(X_train, y_train, X_test, y_test, metrics[i]))
    for i in range(len(rmse)):
        if rmse[i][0] < min:
            min = rmse[i][0]
            best_model = rmse[i][1]
    save_model(best_model)
    rmse.to_csv("rmse.csv")
    best_rmse, beta_rmse = ml_testing_beta(X_test, y_test, best_model)
    beta_rmse.to_csv("beta_rmse.csv")
    return best_model

#save the given model
def save_model(model):
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

#load a previous saved model
def load_model():
    # load the model from disk
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


X_train, X_test, y_train, y_test = create_tuple()
