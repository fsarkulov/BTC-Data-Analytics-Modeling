# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:53:19 2021

@author: farus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
from yahoofinancials import YahooFinancials

import tensorflow as tf
from datetime import timedelta
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV

import keras.initializers
from keras.layers import Dense, Layer, GRU, Dropout, Activation
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers


raw_data = yf.download('BTC-USD',
                       start = '2018-01-01',
                       end = '2021-09-23',
                       progress = False)

raw_data.head()

btc_close = raw_data["Close"]

adf, p ,usedlag, nobs, cvs, aic = sm.tsa.stattools.adfuller(btc_close) 
adf_results_string = 'ADF: {}\np-value: {},\nN: {}, \ncritical values: {}'
print(adf_results_string.format(adf, p, nobs, cvs))


#our p value and ADF value show that we cannot reject the null hypothesis so the data is non-stationary. NOTE however that the returns themselves have some stationarity!

#taking the pacf and plotting
#pacf = sm.tsa.stattools.pacf(btc_close, nlags=30)

#T = len(btc_close)
#sig_test = lambda tau_h: np.abs(tau_h) > 2.58/np.sqrt(T)

#for i in range(len(pacf)):
    #if sig_test(pacf[i]) == False:
       # n_steps = i-1
        #print('n_steps set to', n_steps)
       # break

#this tells us 4 is the max lag until the pacf says there is no correlation. 4 lags is very heavy computationally however

#plt.plot(pacf, label='pacf')
#plt.plot([2.58/np.sqrt(T)]*30, label='99% confidence interval (upper)')
#plt.plot([-2.58/np.sqrt(T)]*30, label='99% confidence interval (lower)')
#plt.xlabel('number of lags')
#plt.legend();
    
#our pacf plot shows that the optimum lag is closer to two than four, to reduce computational intensity we will assume lag of 2 will work fine enough for our desires


#split into train and test
train_ratio = 0.8
split = int(len(btc_close)*train_ratio)

train_df = btc_close.iloc[:split]
test_df = btc_close.iloc[split:]


#scale and standardize
mu = np.float(train_df.mean())
sd = np.float(train_df.std())

standardize = lambda x: (x-mu)/sd

train_df = train_df.apply(standardize)
test_df = test_df.apply(standardize)

#We create overlapping time subintervals to have one-step ahead time prediction. Intuitively, the formula for y_t is dependent on y_t-1 so the subsequences create this correlation by overlapping the subintervals

def lag_my_features(df, n_steps, n_steps_ahead):
    lag_list = []
    for lag in range(n_steps+n_steps_ahead-1,n_steps_ahead-1, -1):
        lag_list.append(df.shift(lag))
        lag_array = np.dstack([i[n_steps_ahead+n_steps-1:] for i in lag_list])

        lag_array = np.swapaxes(lag_array,1,-1)
        return lag_array

#apply lag features to our test and train set
n_steps_ahead = 10   # forecasting horizon
n_steps =4

x_train = lag_my_features(train_df, n_steps, n_steps_ahead)
y_train = train_df.values[n_steps + n_steps_ahead-1:]
y_train_timestamps = train_df.index[n_steps + n_steps_ahead-1:]


x_test = lag_my_features(test_df, n_steps, n_steps_ahead)
y_test = test_df.values[n_steps + n_steps_ahead-1:]
y_test_timestamps = test_df.index[n_steps + n_steps_ahead-1:]
print([tensor.shape for tensor in (x_train, y_train, x_test, y_test)])

#transpose to amke the dimensions match for model fitting 

x_test = np.transpose(x_test)
x_train = np.transpose(x_train)

print([tensor.shape for tensor in (x_train, y_train, x_test, y_test)])


model = Sequential()
model.add(GRU(50,return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(GRU(100, return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "linear"))
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs=50, batch_size=32)
in_samp_pred = model.predict(x_train)
in_samp_compare = [[y_train],[in_samp_pred]]
prediction = model.predict(x_test)
compare_tens = [[y_test],[prediction]]

in_samp_loss = mean_squared_error(y_train, in_samp_pred)
out_samp_loss = mean_squared_error(y_test, prediction)

ind = list(range(0,259))
ind2= list(range(0,1073))
fig, ax = plt.subplots()
ax.plot(ind, y_test, label = 'True BTC Price')
ax.plot(ind, prediction, label = 'Predicted BTC Price')
ax.legend(loc='upper right')
plt.title('GRU Predicted BTC Prices VS True BTC Prices (Out-of-Sample)')
plt.xlabel('Index of Days (01-2021 to 09-2021)')
plt.ylabel('Scaled BTC Closing Price')
plt.show()

fig, ax = plt.subplots()
ax.plot(ind2,y_train , label = 'True BTC Price')
ax.plot(ind2, in_samp_pred, label  = 'Predicted BTC Price')
ax.legend(loc='upper right')
plt.title('GRU Predicted BTC Prices VS True BTC Prices (In Sample)')
plt.xlabel('Index of Days (01-2018 to 01-2021)')
plt.ylabel('Scaled BTC Closing Price')
plt.show()

