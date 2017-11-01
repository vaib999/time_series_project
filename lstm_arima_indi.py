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


p_id = int(input("Type product no. from 0 to 99 "))

predictions =[]

# load dataset
data = pd.read_excel("product_distribution_training_set.xlsx",header = None)

series = data.ix[p_id,1:]
for i in range(2):
        actual_value = series.values
        actual_value = actual_value.tolist()
        
        for i in range(len(predictions)):
                actual_value.append(predictions[i])
        actual_value = numpy.asarray(actual_value)
        
        #[  7.2    8.51   7.65   8.38   8.85   9.55  10.99  10.66  11.24   9.38]
        
        #Differencing to make data stationary
        # Making Stationary series
        interval = 1
        diff = list()
        for i in range(interval, len(data)):
                value = actual_value[i] - actual_value[i - interval]
                diff.append(value)
        diff_values = Series(diff)

        # Features for the supervised problem
        # Making features for supervised learning
        lag = 7
        df = DataFrame(diff_values)
        cols = []
        for i in range(1, lag+1):
                cols.append(df.shift(i))
        cols.append(df)
        df = concat(cols, axis=1)
        df.fillna(0, inplace=True)
        shifted_values = df

        supervised_values = shifted_values.values

        # Splitting data
        train, test = supervised_values[0:-15], supervised_values[-15:]

        # Normalize train and test data in interval [1,-1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train)
        # Normalize train data
        train = train.reshape(train.shape[0], train.shape[1])
        train_norm = scaler.transform(train)
        # Normalize test data
        test = test.reshape(test.shape[0], test.shape[1])
        test_norm = scaler.transform(test)

        # Model Fit
        batch_size = 1
        epochs = 10
        neurons = 4
        X, y = train_norm[:, 0:-1], train_norm[:, -1]
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
        train_reshaped = train_norm[:, 0:-1].reshape(len(train_norm), 1, 7)

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
        
##predictions = predictions.tolist()
##print("LSTM model prediction")
##print("29 days values")
##print(predictions)
#plt.plot(numpy.arange(118),series, color='blue')
#plt.plot(numpy.arange(118,148),predictions, color='red')
#plt.show()

#####################################################################################################################################
#ARIMA MODEL
 
# invert differenced value
def revert_difference(past, y_pred, interval=1):
	return y_pred + past[-interval]

# load dataset
data = pd.read_excel("product_distribution_training_set.xlsx",header = None)
actual_value = data.ix[p_id,1:]
actual_value = actual_value.astype(float)
# seasonal difference
X = actual_value.values
days_taken = 117

# create a differenced series
diff = []
for i in range(days_taken, len(X)):
        value = X[i] - X[i - days_taken]
        diff.append(value)
difference = numpy.array(diff)

# fit model
#The parameters are changed for different series to get better results
model = ARIMA(X, order=(2,1,1))
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
