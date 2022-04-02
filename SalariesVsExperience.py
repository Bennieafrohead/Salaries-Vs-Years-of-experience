
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the datasets
dataset = pd.read_csv("C:/Users/Hp/Documents/Data analytics/Everything Python/Python/4. Regression Simple Linear Regression/3.3 Salaries.csv.csv")
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values

# splitting data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1/3, random_state=0) 

#fitting the linear regression to train the datasets
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
regressor.fit(X_train, Y_train)

#predicting the values of test sets
X_test = X_test.reshape(-1, 1)
Y_pred = regressor.predict(X_test)

#visualizing the result of the trained data sets
plt.scatter(X_train, Y_train)
plt.plot(X_train, regressor.predict(X_train),color='red')
plt.title('Salaries Vs Experience (Training Set Results)')
plt.xlabel('Years of experience')
plt.ylabel('Salaies')
plt.show()

#visualizing the result of the test data sets
plt.scatter(X_test, Y_test)
plt.plot(X_train, regressor.predict(X_train),color='red')
plt.title('Salaries Vs Experience (Testing Set Results)')
plt.xlabel('Years of experience')
plt.ylabel('Salaies')
plt.show()