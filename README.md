# Netflix-Stock-Price-Prediction
 Netflix Stock PricePrediction using Time Series
 
# Overview
This project involves analyzing and predicting the stock prices of Netflix (NFLX) using historical data and various machine learning techniques. The goal is to build a model that can forecast future stock prices based on past performance, providing valuable insights for investors and stakeholders. The dataset includes daily stock prices of Netflix, capturing essential attributes such as Open, High, Low, Close, Volume, and Date.

# About Dataset
The Dataset contains data for 5 years ie. from 5th Feb 2018 to 5th Feb 2022

The art of forecasting stock prices has been a difficult task for many of the researchers and analysts. In fact, investors are highly interested in the research area of stock price prediction. For a good and successful investment, many investors are keen on knowing the future situation of the stock market. Good and effective prediction systems for the stock market help traders, investors, and analyst by providing supportive information like the future direction of the stock market.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Loading the Dataset

df=pd.read_csv("/NFLX.csv")

**Exploratory** **data** **analysis**

Check top five rows 

df.head()

**Bottom** **five rows**

df.tail()

**create deep copy object**

it means deep copy object will not effect the original object

viz=df.copy()



**checking missing values**



df.isna().sum()



**checking no.of rows and columns**

df.shape

**checking dataset information**

df.info()

**checking five number summary**

df.describe()

**Data preparation**

X=df.drop(["Close","Date","Adj Close"],axis=1)
y=df["Close"]

**splitting the data**

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

X_train.head()

y_train.head()

**model creation**

model=LinearRegression()

**model training**

model.fit(X_train,y_train)

**model prediction**

y_pred=model.predict(X_test)

y_pred

**check accuracy**

model.score(X_train,y_train)

model.score(X_test,y_test)

result=model.predict([[262.000000, 267.899994, 250.029999, 11896100]])

result

**check cost function(error)**

print("mse is:",mean_squared_error(y_test,y_pred))
print("mae is:",mean_absolute_error(y_test,y_pred))
print("mape is:",mean_absolute_percentage_error(y_pred,y_test))
print("rmse is:",np.sqrt(mean_squared_error(y_pred,y_test)))

def style():
    plt.figure(facecolor='black', figsize=(15,6))
    ax = plt.axes()

    ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to white
    ax.tick_params(axis='y', colors='white')    #setting up Y-axis tick color to white

    ax.spines['left'].set_color('white')        #setting up Y-axis spine color to white
    
    ax.spines['bottom'].set_color('white')      #setting up X-axis spine color to white

    ax.set_facecolor("black")                   # Setting the background color of the plot using set_facecolor() method

viz.head()

**converting object into datetime datatype**

viz["Date"]=pd.to_datetime(viz["Date"])

viz.dtypes

**creating new dataframe**

df2=pd.DataFrame({"date":viz["Date"],"close":viz["Close"]})
df2.set_index("date",inplace=True)

df2.head()

df2.shape

df2=df2.asfreq('D')

df2.head()

df2.shape

**Data Vizualization**

df2.plot()
plt.show()

style()

plt.title("Closing Stock price",color="white")
plt.plot(viz["Date"],viz["Close"],color="red")
plt.legend(["close"],loc="upper left",facecolor="black",labelcolor="white")
plt.show()

style()

plt.scatter(y_pred, y_test, color='red', marker='o')
plt.scatter(y_test, y_test, color='blue')
plt.plot(y_test, y_test, color='lime')

close=model.predict(X)

df3=pd.DataFrame({"actual":df["Close"],"predicted":close})

df3.describe()

df3["Date"]=viz["Date"]

df3.head()

df3.set_index(["Date"],inplace=True)

df3.head()

df3.to_csv("output_stock.csv")
