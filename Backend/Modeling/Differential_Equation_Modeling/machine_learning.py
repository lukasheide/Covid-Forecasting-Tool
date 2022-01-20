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

    # 2a) extend beta with 1 row, so including beta_t-1 is easier
    beta_t_1 = pd.DataFrame(beta)
    beta_t_1.loc[0] = [0]  # adding a row
    beta_t_1.index = beta_t_1.index + 1  # shifting index
    beta_t_1 = beta_t_1.sort_index()  # sorting by index
    beta_t_1 = beta_t_1.drop(beta_t_1.index[-1:])
    beta_t_1.rename(columns={'beta': 'beta_t_1'}, inplace=True)

    # 2b) extend beta with 2 row, so including beta_t-2 is easier
    beta_t_2 = pd.DataFrame(beta_t_1) # creating be
    beta_t_2.loc[0] = [0] # adding a row
    beta_t_2.index = beta_t_2.index + 1  # shifting index
    beta_t_2 = beta_t_2.sort_index()  # sorting by index
    beta_t_2 = beta_t_2.drop(beta_t_2.index[-1:])
    beta_t_2.rename(columns={'beta_t_1': 'beta_t_2'}, inplace=True)

    # 3)insert a new column in a dataframe:
    df = pd.concat([df, beta_t_1], axis=1)
    df = pd.concat([df, beta_t_2], axis=1)

    #delte infections = 0, fill mean in empty temperature/wind cells
    df = delete_zero_infections(df)
    fill_empty_rows(df)

    #delete every first and second week
    df.drop(df[df['week'] == 1].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.drop(df[df['week'] == 2].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

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
#X_test = pd.read_csv("x_test.csv")
#X_train = pd.read_csv("x_train.csv")
#y_test = pd.read_csv("y_test.csv")
#y_train = pd.read_csv("y_train.csv")
fitted_model = run_all(X_train, y_train, X_test, y_test)
predict_tuple = [0,1,0,83.92999999999999,-14.0,0.3999999999999999,8.642857142857142,125.69387755102044,0.0447375155108553,0.0899554552236932]
fitted_model = load_model()
print(predict_beta(fitted_model, predict_tuple))
