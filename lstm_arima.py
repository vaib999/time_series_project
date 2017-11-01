from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib.pylab as plt
import numpy
from statsmodels.tsa.arima_model import ARIMA

####################################################################################################################################
#LSTM MODEL

# Getting back normalized values
def norm_back(norm, X, value):
        add = []
        for x in X:
                add.append(x)

        new_row = add + [y_pred]
        array = numpy.array(new_row)
        array = array.reshape(1, len(array))
        invert_mat = norm.inverse_transform(array)
        return invert_mat[0, -1]
 

#Predicting every day value
def predict_model(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	y_pred = model.predict(X, batch_size=batch_size)
	return y_pred[0,0]


predictions =[]

# load dataset
data = pd.read_excel("product_distribution_training_set.xlsx",header = None)

#Taking mean of each product sale for every day
series = data.ix[:,1:].mean()

for i in range(2):
        actual_value = series.values
        #[  7.2    8.51   7.65   8.38 .......]
        actual_value = actual_value.tolist()
        #[7.2, 8.51, 7.65, 8.38, 8.85, .......]
        
        for i in range(len(predictions)):
                actual_value.append(predictions[i])
                
        actual_value = numpy.asarray(actual_value)
        #[  7.2    8.51   7.65   8.38   8.85....]
        
        #Differencing to make data stationary
        # Making Stationary series
        
        interval = 1
        diff = list()
        
        for i in range(interval, len(data)):
                diff.append(actual_value[i] - actual_value[i - interval])
                
        diff_values = Series(diff)
        ##0     1.31
        ##1    -0.86
        ##2     0.73
        ##3     0.47
        ##4     0.70
        ##5     1.44
        ##6    -0.33

        # Features for the supervised problem
        # Making features for supervised learning
        
        lag = 30
        df = DataFrame(diff_values)
        ##       0
        ##0   1.31
        ##1  -0.86
        ##2   0.73
        ##3   0.47
        ##4   0.70
        ##5   1.44
        cols = []
        for i in range(1, lag+1):
                cols.append(df.shift(i))
        cols.append(df)
        ##[       0
        ##0    NaN
        ##1   1.31
        ##2  -0.86
        ##3   0.73
        ##4   0.47
        ##5   0.70
        ##6   1.44
        ##7  -0.33...]
        df = concat(cols, axis=1)
        ##       0     0     0     0     0     0     0      0      0      0  ...   \
        ##0    NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN    NaN  ...    
        ##1   1.31   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN    NaN  ...    
        ##2  -0.86  1.31   NaN   NaN   NaN   NaN   NaN    NaN    NaN    NaN  ...    
        ##3   0.73 -0.86  1.31   NaN   NaN   NaN   NaN    NaN    NaN    NaN  ...    
        ##4   0.47  0.73 -0.86  1.31   NaN   NaN   NaN    NaN    NaN    NaN  ...    
        ##5   0.70  0.47  0.73 -0.86  1.31   NaN   NaN    NaN    NaN    NaN  ...    
        ##6   1.44  0.70  0.47  0.73 -0.86  1.31   NaN    NaN    NaN    NaN  ...    
        ##7  -0.33  1.44  0.70  0.47  0.73 -0.86  1.31    NaN    NaN    NaN  ...    
        ##8   0.58 -0.33  1.44  0.70  0.47  0.73 -0.86   1.31    NaN    NaN  ...    
        ##9  -1.86  0.58 -0.33  1.44  0.70  0.47  0.73  -0.86   1.31    NaN  ...    
        ##10  0.87 -1.86  0.58 -0.33  1.44  0.70  0.47   0.73  -0.86   1.31  ...  
        df.fillna(0, inplace=True)
        shifted_values = df

        supervised_values = shifted_values.values
        ##[[ 0.    0.    0.   ...,  0.    0.    1.31]
        ## [ 1.31  0.    0.   ...,  0.    0.   -0.86]
        ## [-0.86  1.31  0.   ...,  0.    0.    0.73]

        # Splitting data
        train, test = supervised_values[0:-15], supervised_values[-15:]
        ##print(train)
        ##[[ 0.    0.    0.   ...,  0.    0.    1.31]
        ## [ 1.31  0.    0.   ...,  0.    0.   -0.86]
        ## [-0.86  1.31  0.   ...,  0.    0.    0.73]
        ##print(test)
        ##[[ -2.05  -2.49   0.52   3.77   1.85  -1.91  -4.06  -2.15   2.28  -2.18
        ##    6.67  -2.9   -1.31  -1.21   0.74   0.44  -6.44   5.45  -1.79  -2.67
        ##   -2.4    1.05  11.97  -2.46   1.55  -6.56  -0.11  -4.5    1.72  -1.02
        ##   -1.65]
        # Normalize train and test data in interval [1,-1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # Normalize train data
        train = train.reshape(train.shape[0], train.shape[1])
        train_norm = scaler.transform(train)
        ##print(train_norm)
        ##[[-0.29195899 -0.29195899 -0.29195899 ..., -0.16716123 -0.16716123
        ##  -0.15056665]
        ## [-0.15056665 -0.29195899 -0.29195899 ..., -0.16716123 -0.16716123
        ##  -0.38478144]
        ## [-0.38478144 -0.15056665 -0.29195899 ..., -0.16716123 -0.16716123
        ##  -0.21316784]        
        # Normalize test data
        test = test.reshape(test.shape[0], test.shape[1])
        test_norm = scaler.transform(test)
        ##print(test_norm)
        ##[[ -5.13221802e-01  -5.60712358e-01  -2.35833783e-01   1.14948732e-01
        ##   -9.22827847e-02  -4.98111171e-01  -7.30167296e-01  -5.24015111e-01
        ##   -4.58715596e-02  -5.27253103e-01   4.27954668e-01  -6.04964922e-01
        ##   -4.33351322e-01  -4.22558014e-01  -2.12088505e-01  -2.44468430e-01
        ##   -9.87048030e-01   2.96276309e-01  -4.85159201e-01  -5.80140313e-01
        ##   -5.50998381e-01  -1.78629250e-01   1.97431782e+00  -3.41894061e-01
        ##    3.01765650e-01  -1.39615385e+00  -1.55769231e-01  -1.05736894e+00
        ##    1.73095945e-01  -3.68941642e-01  -4.70048570e-01]

        # Model Fit
        batch_size = 1
        epochs = 1
        neurons = 4
        
        X, y = train_norm[:, 0:-1], train_norm[:, -1]
        ##print(X)
        ##[[-0.29195899 -0.29195899 -0.29195899 ..., -0.16716123 -0.16716123
        ##  -0.16716123]
        ## [-0.15056665 -0.29195899 -0.29195899 ..., -0.16716123 -0.16716123
        ##  -0.16716123]
        ## [-0.38478144 -0.15056665 -0.29195899 ..., -0.16716123 -0.16716123
        ##  -0.16716123]
        ##print(y)
        ##[-0.15056665 -0.38478144 -0.21316784 -0.24123044 -0.21640583 -0.13653535
        ## -0.3275769  -0.2293578  -0.49271452 -0.1980572  -0.16243929 -0.28656233
        ## -0.1699946  -0.43119266  0.00377766 -0.42471668 -0.1279007   0.06206152
        ## -0.44414463 -0.53588775  0.00701565 -0.43658931 -0.39233675 -0.12574204
        ##  0.22611981 -0.48839719 -0.0998381  -0.21964382 -0.52077712 -0.41176471
        ## -0.16135996 -0.28980032 -0.54344307  0.07933081  0.05450621 -0.285483
        ## -0.66540745 -0.0577442  -0.01780896 -0.39341608  0.07285483 -0.51214247
        ## -0.4991905  -0.05126821 -0.3729088  -0.22611981 -0.10091743 -0.32002159
        ## -0.74635726 -0.24986508 -0.21748516 -0.54668106  0.3448462  -0.23259579
        ## -0.40205073 -0.10631409 -0.77765785 -0.30383162 -1.         -0.12466271
        ## -0.55747437  1.         -0.17862925 -0.55099838 -0.58014031 -0.4851592
        ##  0.29627631 -0.98704803 -0.24446843 -0.21208851 -0.42255801 -0.43335132
        ## -0.60496492  0.42795467 -0.5272531  -0.04587156 -0.52401511 -0.7301673
        ## -0.49811117 -0.09228278  0.11494873 -0.23583378 -0.56071236 -0.5132218 ]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        for i in range(epochs):
                model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
                model.reset_states()
        lstm_model = model

        # Predicting for state building
        train_reshaped = train_norm[:, 0:-1].reshape(len(train_norm), 1, 30)

        lstm_model.predict(train_reshaped, batch_size=1)

        # Predicting in group of 15
        for i in range(len(test_norm)):
                X, y = test_norm[i, 0:-1], test_norm[i, -1]
                y_pred = predict_model(lstm_model, 1, X)
                # Norm back
                y_pred = norm_back(scaler, X, y_pred)
                
                # Getting back the original value
                y_pred = y_pred + actual_value[-(len(test_norm)+1-i)]
                
                y_pred = numpy.asarray(int(y_pred))
                if y_pred<0:
                        y_pred = 0
                predictions = numpy.append(predictions,y_pred)
        
predictions = predictions.tolist()
##print(predictions)
##[15.0, 13.0, 11.0, 14.0, 13.0, 10.0, 8.0, 10.0, 12.0,.........]

##print("LSTM model prediction")
##print("29 days values")
##print(predictions)
##plt.plot(numpy.arange(118),series, color='blue')
##plt.plot(numpy.arange(118,148),predictions, color='red')
##plt.show()

#####################################################################################################################################
#ARIMA MODEL
 
# invert differenced value
def revert_difference(past, y_pred, interval=1):
	return y_pred + past[-interval]

# load dataset
data = pd.read_excel("product_distribution_training_set.xlsx",header = None)
actual_value = data.ix[:,1:].mean()
actual_value = actual_value.astype(float)
# seasonal difference
X = actual_value.values
days_taken = 7

# create a differenced series
diff = []
for i in range(days_taken, len(X)):
        value = X[i] - X[i - days_taken]
        diff.append(value)
difference = numpy.array(diff)

# fit model
model = ARIMA(X, order=(10,1,1))
model_fit = model.fit(disp=0)
# multi-step out-of-sample forecast
starting = len(difference)
ending = starting + 29
predicted = model_fit.predict(start=starting, end=ending)
# invert the differenced forecast to something usable
past = [x for x in X]
invert = []
for y_pred in predicted:
	original = revert_difference(past, y_pred, days_taken)
	invert.append(int(original))
	past.append(original)
    
#plt.plot(numpy.arange(118),actual_value, color='blue')
#plt.plot(numpy.arange(118,148),invert, color='red')
#plt.show()

##print("ARIMA Prediction")
##print(invert)

#####################################################################################################################################
#ENSEMBLING

average = []
for i in range(len(invert)):
        average.append((invert[i]+predictions[i])//2)
"""
for i in range(len(average)):
        if average[i]<0:
                average[i] = 0
"""
##print("Overall ensemble prediction")
##print(average)
#plt.plot(numpy.arange(118),series, color='blue')
#plt.plot(numpy.arange(118,148),average, color='red')
#plt.show()
