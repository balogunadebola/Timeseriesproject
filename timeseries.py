#import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

#load timeseries data into a dataframe
data = pd.read_csv("C:/Users/Akorede Balogun/Downloads/Sales.csv", parse_dates=['Date'], index_col='Date')

#Display the first few rows
print(data.head())

#Extract relevant column if there is more than one
data = data['Sales']

#Plot the time series
plt.figure(figsize=(10,6))
plt.plot(data, label='Time Series Data')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

#split data into training and test sets
train_size = int(len(data)* 0.8) #80% training data
train, test = data[:train_size], data[train_size:]

#Plot train and test sets
plt.figure(figsize=(10,6))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data', color='red')
plt.title('Train and Test Split')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

#Build and fit an ARIMA model
model = ARIMA(train, order=(5,1,0)) #(p d q) parameters

#fit the model 
model_fit = model.fit()

#Print the model summary
print(model_fit.summary())

#Make predictions
predictions = model_fit.forecast(steps=len(test))

#Plot the predictions against the test data
plt.figure(figsize=(10,6))
plt.plot(test.index, test, label='Actual Data', color='red')
plt.plot(test.index, predictions, label='Predicted Data', color='green')
plt.title('Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

#Calclulate the mean squared error
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')