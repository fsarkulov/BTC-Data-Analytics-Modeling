# BTC-Data-Analytics-Modeling
This is a project that uses historical BTC time series data to train and test various models. Along with price prediction, a theoretical exercise was done to see how the Black Scholes Merton and Heston Options pricing formulas compare to the true options prices taken from Derebit. 
The R file contains a preliminary analysis of the BTC time series data taken from Yahoo Finanace. The data is compared against a normal distribution to get a sense of the probability distribution and derive various statistics like mean, skewness, and kurtosis. The data is also tested for stationarity using the ADF test and the Hurst Exponent. The second part of the R file applies stochastic volatility process using Markov Chain Monte Carlo sampling. The results provide various statistics needed to implement the Heston Options Pricing Model. The results are compared against the true options prices and the Black Scholes Merton derived options prices. The final part of the R file implements a NN-LSTM model to predict BTC closing prices. 
The Python file implements a NN-GRU model to predict BTC closing prices. 
The PDF report is a complete report on the project and the findings. 
