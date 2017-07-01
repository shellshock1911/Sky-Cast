#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 22:56:52 2017

@author: BennyBluebird
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

def get_data(airline, airport, category):
    
    series = pd.read_csv('./aviation_data/{}-{}.csv'.format(airline, airport), index_col='Date', 
                    parse_dates=True, date_parser=dateparse)

    data = series['{}_Domestic'.format(category)].astype(np.float64)
    
#   TODO: Add ability to create and return dataframe of multiple categories
    
    return data

def test_stationarity(time_series):

    moving_avg = time_series.rolling(window=12).mean()
    moving_std = time_series.rolling(window=12).std()
    name = time_series.name.split('_')[0]
    
    plt.plot(time_series, color='blue', label='Monthly {}'.format(name))
    plt.plot(moving_avg, color='red', label='Moving Average')
    plt.plot(moving_std, color='black', label='Moving Std.')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    print('Results of Dickey-Fuller Test:')
    test = adfuller(time_series, autolag='AIC')
    test_output = pd.Series(test[0:4], index=['Test Statistic', 'p-value',
                         '#Lags Used', 'Number of Observations Used'])
    for key, value in test[4].items():
        test_output['Critical Value {}'.format(key)] = value
    print(test_output)
    
def difference(dataset, lag=1):
    
    difference = list()
    for i in range(lag, len(dataset)):
        value = dataset[i] - dataset[i - lag]
        difference.append(value)
    return np.array(difference)

def revert_difference(history, pred, lag=1):
    return pred + history[-lag]

def predict_final_year(time_series, order=(12,1,2), search=False):
    
#   TODO: Make it possible to lag to varying distances in the past
#         and forecast to varying distances in the future
    
    data = time_series.values
    train, test = data[:-12], data[-12:]
    differenced = difference(train, lag=12)
    model = ARIMA(differenced, order=order)
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(12)[0]
    history = [x for x in train]
    for pred in forecast:
        reverted = revert_difference(history, pred, lag=12)
        history.append(reverted)
    preds = np.array(history[-12:])
    
    if search:
        return mean_squared_error(test, preds)
    
    print 'RMSE: ' + str(np.sqrt(mean_squared_error(test, preds)))
    print 'R_SQ: '+ str(r2_score(test, preds))
    plt.plot(test)
    plt.plot(preds, color='red')
    plt.show()
    
def grid_search(dataset, p_values=range(13), d_values=range(3), q_values=range(3)):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    print 'Testing ARIMA: {}'.format(order)
                    mse = predict_final_year(dataset, order, search=True)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print 'MSE: {:.3f}\n'.format(mse)
                except:
                    continue
    print 'Best ARIMA: {}'.format(best_cfg)
    print 'Best RMSE: {}'.format(np.sqrt(best_score)) 