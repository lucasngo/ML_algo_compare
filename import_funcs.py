import dash
import plotly.graph_objs as go
import pandas as pd 
import numpy as np
import math
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

def preprocessing(df):
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
    df['hl_pct'] = (df['high'] - df['low']) / df['close'] * 100.0
    df['pct_change'] = (df['close'] - df['open']) / df['open'] * 100.0
    df = df.drop(['adj_close'], axis = 1)
    df_y = df['close']
    df_x = df.drop(['close'], axis = 1)
    train_size = math.ceil(0.8*len(df_x))
    test_size = len(df_x) - train_size
    x_train, y_train, x_test, y_test = df_x[:train_size], df_y[:train_size], df_x[train_size:], df_y[train_size:]
    return(x_train, y_train, x_test, y_test)

def preprocessing_arima(df):
    df = df.reset_index()
    
    # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
    # Add high low percentage and percent change columns
    df['hl_pct'] = (df['high'] - df['low']) / df['close'] * 100.0
    df['pct_change'] = (df['close'] - df['open']) / df['open'] * 100.0
    
    # Convert Date column to datetime
    df.loc[:, 'date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
    
    # Dropping adjusted close     
    df = df.drop(['adj_close'], axis = 1)
    
    # Sort by datetime
    df.sort_values(by='date', inplace=True, ascending=True)
    data = df.loc[:, ['date', 'close']]
    train_size = math.ceil(0.8*len(data))
    test_size = len(data) - train_size
    train = data[:train_size]
    test = data[train_size:]
    return(train, test)   

def RMSE(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return math.sqrt(mean_squared_error(y_true, y_pred))

def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_arima_pred(train, test, p=5, q=1, d=0):
    history = [x for x in train['close']]
    model_predictions = []
    N_test_observations = len(test)

    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(p,q,d))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test.iloc[time_point]['close']
        history.append(true_test_value)
    
    test['forecast'] = model_predictions
    test.index = test['date']
    return test

def get_lr_pred(x_train, y_train, x_test, y_test, fit_intercept=True, norm=True):
    lr_model = LinearRegression(fit_intercept=fit_intercept, normalize=norm)
    lr_model.fit(x_train, y_train)
    index = y_test.index
    predictions = pd.Series(lr_model.predict(x_test), index = index)
    return(predictions)

def get_knn_pred(x_train, y_train, x_test, y_test, neighbors=15, p=1, weights='distance'):
    knn = KNeighborsRegressor(p=p, weights=weights, n_neighbors=neighbors)
    
    # scaling the features (between a value of 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)
    x_test_scaled = scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_test_scaled)
    knn.fit(x_train, y_train)
    index = y_test.index
    predictions = pd.Series(knn.predict(x_test), index = index)
    return(predictions)

def get_svr_pred(x_train, y_train, x_test, y_test, kernel = 'linear', c = 2, eps = 0.3):
    svr = SVR(C=c, epsilon = eps, kernel = kernel)
    
    # scaling the features (between a value of 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)
    x_test_scaled = scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_test_scaled)
    svr.fit(x_train, y_train)
    index = y_test.index
    predictions = pd.Series(svr.predict(x_test), index = index)
    return(predictions)

def get_xgb_pred(x_train, y_train, x_test, y_test, min_child_weight=3, max_depth=5, learn_rate=0.1):
    xgbr = xgb.XGBRegressor(min_child_weight=min_child_weight, max_depth=max_depth,learning_rate= learn_rate,verbosity=0)
    xgbr.fit(x_train, y_train)
    index = y_test.index
    predictions = pd.Series(xgbr.predict(x_test), index = index)
    return(predictions)

def get_plots(algo, y_true, y_pred):
    if algo == 'ARIMA':
        fig = go.Figure()
        fig.add_trace(go.Scatter(
        x=y_pred.index,
        y=y_pred[1:]['forecast'],
        name = 'Forecast'))
        fig.add_trace(go.Scatter(
        x=y_pred.index,
        y=y_pred[1:]['close'],
        name = 'Actual'))
        fig.update_layout(title='Actual V/S Predicted: ' + algo,
                   xaxis_title='Date',
                   yaxis_title='Close Price')
        return fig
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
        x=y_true.index,
        y=y_true,
        name = 'Actual'))
        fig.add_trace(go.Scatter(
        x=y_pred.index,
        y=y_pred,
        name = 'Forecast'))
        fig.update_layout(title='Actual V/S Predicted: ' + algo,
                   xaxis_title='Date',
                   yaxis_title='Close Price')
        return fig

def cum_se(algo_1, algo_2, y_true_1, y_pred_1, y_true_2, y_pred_2):
    errors_algo_1 = (y_true_1 - y_pred_1)**2
    cum_errors_1 = np.cumsum(errors_algo_1)
    errors_algo_2 = (y_true_2 - y_pred_2)**2
    cum_errors_2 = np.cumsum(errors_algo_2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=cum_errors_1.index,
    y=cum_errors_1,
    name = algo_1))
    fig.add_trace(go.Scatter(
    x=cum_errors_2.index,
    y=cum_errors_2,
    name = algo_2))
    fig.update_layout(title='Comparison Between: '+algo_1+' V/S '+algo_2,
                xaxis_title='Date',
                yaxis_title='Close Price')
    return fig

def get_output(algo_1, algo_2, metric, stock_data, values1, values2):
    df = stock_data
    rmse = []
    mape = []
    print(values1, values2)

    # checking which metric is being used    
    if metric == 'RMSE' :
        # preprocessing data and applying the respective model for algo 1
        if algo_1 == 'ARIMA':
            values = list(map(int, values1))
            print(values)
            y_pred_1 = get_arima_pred(preprocessing_arima(df)[0],preprocessing_arima(df)[1], values[0], values[1], values[2])
            y_true_1 = preprocessing_arima(df)[1]
            rmse_val_1 = RMSE(y_true_1[1:]['close'], y_pred_1[1:]['forecast'])
            rmse.append(rmse_val_1)
            fig1 = get_plots(algo_1, y_true_1, y_pred_1)
        elif algo_1 == 'LR':
            values = [i == 'True' for i in values1]
            print(values)
            y_pred_1 = get_lr_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[0], values[1])
            y_true_1 = preprocessing(df)[3]
            rmse_val_1 = RMSE(y_true_1, y_pred_1)
            rmse.append(rmse_val_1)
            fig1 = get_plots(algo_1, y_true_1, y_pred_1)
        elif algo_1 == 'KNN':
            weights = values1[1]
            print(weights)
            values = list(map(int, [values1[0], values1[2]]))
            print(values)
            y_pred_1 = get_knn_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[0], weights, values[1])
            y_true_1 = preprocessing(df)[3]
            rmse_val_1 = RMSE(y_true_1, y_pred_1)
            rmse.append(rmse_val_1)
            fig1 = get_plots(algo_1, y_true_1, y_pred_1)
        elif algo_1 == 'SVR':
            kernel = values1[0]
            print(kernel)
            values = list(map(int, values1[1:]))
            print(values)
            y_pred_1 = get_svr_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], kernel, values[0], values[1])
            y_true_1 = preprocessing(df)[3]
            rmse_val_1 = RMSE(y_true_1, y_pred_1)
            rmse.append(rmse_val_1)
            fig1 = get_plots(algo_1, y_true_1, y_pred_1)
        elif algo_1 == 'XGB':
            values = list(map(int, values1))
            print(values)
            y_pred_1 = get_xgb_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[2], values[0], values[1])
            y_true_1 = preprocessing(df)[3]
            rmse_val_1 = RMSE(y_true_1, y_pred_1)
            rmse.append(rmse_val_1)
            fig1 = get_plots(algo_1, y_true_1, y_pred_1)

        # preprocessing data and applying the respective model for algo 2
        if algo_2 == 'ARIMA':
            values = list(map(int, values2))
            print(values)
            y_pred_2 = get_arima_pred(preprocessing_arima(df)[0],preprocessing_arima(df)[1], values[0], values[1], values[2])
            y_true_2 = preprocessing_arima(df)[1]
            rmse_val_2 = RMSE(y_true_1[1:]['close'], y_pred_2[1:]['forecast'])
            rmse.append(rmse_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        elif algo_2 == 'LR':
            values = [i == 'True' for i in values2]
            print(values)
            y_pred_2 = get_lr_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[0], values[1])
            y_true_2 = preprocessing(df)[3]
            rmse_val_2 = RMSE(y_true_2, y_pred_2)
            rmse.append(rmse_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        elif algo_2 == 'KNN':
            weights = values2[1]
            print(weights)
            values = list(map(int, [values2[0], values2[2]]))
            print(values)
            y_pred_2 = get_knn_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[0], weights, values[1])
            y_true_2 = preprocessing(df)[3]
            rmse_val_2 = RMSE(y_true_2, y_pred_2)
            rmse.append(rmse_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        elif algo_2 == 'SVR':
            kernel = values2[0]
            print(kernel)
            values = list(map(int, values2[1:]))
            print(values)
            y_pred_2 = get_svr_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], kernel, values[0], values[1])
            y_true_2 = preprocessing(df)[3]
            rmse_val_2 = RMSE(y_true_2, y_pred_2)
            rmse.append(rmse_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        elif algo_2 == 'XGB':
            values = list(map(int, values2))
            print(values)
            y_pred_2 = get_xgb_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[2], values[0], values[1])
            y_true_2 = preprocessing(df)[3]
            rmse_val_2 = RMSE(y_true_2, y_pred_2)
            rmse.append(rmse_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        fig3 = cum_se(algo_1, algo_2, y_true_1, y_pred_1, y_true_2, y_pred_2)
    else:
        # preprocessing data and applying the respective model for algo 1
        if algo_1 == 'ARIMA':
            values = list(map(int, values1))
            print(values)
            y_pred_1 = get_arima_pred(preprocessing_arima(df)[0],preprocessing_arima(df)[1], values[0], values[1], values[2])
            y_true_1 = preprocessing_arima(df)[1]
            mape_val_1 = MAPE(y_true_1[1:]['close'], y_pred_1[1:]['forecast'])
            mape.append(mape_val_1)
            fig1 = get_plots(algo_1, y_true_1,y_pred_1)
        elif algo_1 == 'LR':
            values = [i == 'True' for i in values1]
            print(values)
            y_pred_1 = get_lr_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[0], values[1])
            y_true_1 = preprocessing(df)[3]
            mape_val_1 = MAPE(y_true_1, y_pred_1)
            mape.append(mape_val_1)
            fig1 = get_plots(algo_1, y_true_1, y_pred_1)
        elif algo_1 == 'KNN':
            weights = values1[1]
            print(weights)
            values = list(map(int, [values1[0], values1[2]]))
            print(values)
            y_pred_1 = get_knn_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[0], weights, values[1])
            y_true_1 = preprocessing(df)[3]
            mape_val_1 = MAPE(y_true_1, y_pred_1)
            mape.append(mape_val_1)
            fig1 = get_plots(algo_1, y_true_1, y_pred_1)
        elif algo_1 == 'SVR':
            kernel = values1[0]
            print(kernel)
            values = list(map(int, values1[1:]))
            print(values)
            y_pred_1 = get_svr_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], kernel, values[0], values[1])
            y_true_1 = preprocessing(df)[3]
            mape_val_1 = MAPE(y_true_1, y_pred_1)
            mape.append(mape_val_1)
            fig1 = get_plots(algo_1, y_true_1, y_pred_1)
        elif algo_1 == 'XGB':
            values = list(map(int, values1))
            print(values)
            y_pred_1 = get_xgb_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[2], values[0], values[1])
            y_true_1 = preprocessing(df)[3] 
            mape_val_1 = MAPE(y_true_1, y_pred_1)
            mape.append(mape_val_1)
            fig1 = get_plots(algo_1, y_true_1, y_pred_1)
         
        # preprocessing data and applying the respective model for algo 2
        if algo_2 == 'ARIMA':
            values = list(map(int, values2))
            print(values)
            y_pred_2 = get_arima_pred(preprocessing_arima(df)[0],preprocessing_arima(df)[1], values[0], values[1], values[2])
            y_true_2 = preprocessing_arima(df)[1]
            mape_val_2 = MAPE(y_true_2[1:]['close'], y_pred_2[1:]['forecast'])
            mape.append(mape_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        elif algo_2 == 'LR':
            values = [i == 'True' for i in values2]
            print(values)
            y_pred_2 = get_lr_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[0], values[1])
            y_true_2 = preprocessing(df)[3]
            mape_val_2 = MAPE(y_true_2, y_pred_2)
            mape.append(mape_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        elif algo_2 == 'KNN':
            weights = values2[1]
            print(weights)
            values = list(map(int, [values2[0], values2[2]]))
            print(values)
            y_pred_2 = get_knn_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[0], weights, values[1])
            y_true_2 = preprocessing(df)[3]
            mape_val_2 = MAPE(y_true_2, y_pred_2)
            mape.append(mape_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        elif algo_2 == 'SVR':
            kernel = values2[0]
            print(kernel)
            values = list(map(int, values2[1:]))
            print(values)
            y_pred_2 = get_svr_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], kernel, values[0], values[1])
            y_true_2 = preprocessing(df)[3]
            mape_val_2 = MAPE(y_true_2, y_pred_2)
            mape.append(mape_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        elif algo_2 == 'XGB':
            values = list(map(int, values2))
            print(values)
            y_pred_2 = get_xgb_pred(preprocessing(df)[0], preprocessing(df)[1], preprocessing(df)[2], preprocessing(df)[3], values[2], values[0], values[1])
            y_true_2 = preprocessing(df)[3]
            mape_val_2 = MAPE(y_true_2, y_pred_2)
            mape.append(mape_val_2)
            fig2 = get_plots(algo_2, y_true_2, y_pred_2)
        fig3 = cum_se(algo_1, algo_2, y_true_1, y_pred_1, y_true_2, y_pred_2)
    return(fig1, fig2, fig3)
