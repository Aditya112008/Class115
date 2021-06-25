import csv 
import pandas as pd 
import plotly.express as px

df = pd.read_csv("./data.csv")

score_list = df["Score"].tolist()
accepted_list = df["Accepted"].tolist()

fig = px.scatter(x = score_list, y = accepted_list)
fig.show()


import numpy as np
score_array = np.array(score_list)
accepted_array = np.array(accepted_list)

#Slope and intercept using pre-build function of numpy
m,c = np.polyfit(score_array,accepted_array,1)

y = []
for x in score_array:
    y_value = m*x + c
    y.append(y_value)

#Plotting the graph 

fig = px.scatter(x = score_array, y = accepted_array)
fig.update_layout(shapes = [
    dict(
        type = 'line',
        y0 = min(y),y1 = max(y),
        x0 = min(score_array),x1 = max(score_array)
    )
])

fig.show()

import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
#We have to install pip3 matplotlib
#Then we need to do pip3 install sklearn 

#Reshape the array using the reshape function 
#Reshapes the array wihhout changing the data.it changes the shape and size of the array 

X = np.reshape(score_list,(len(score_list),1))
Y = np.reshape(accepted_list,(len(accepted_list),1))

#Use Logistic Regression Model to fit the data into the model so that it can make predictions with maximum accuracy 

lr = LogisticRegression()
lr.fit(X,Y)

#Creating a scatter plot 

plt.figure()
plt.scatter(X.ravel(),Y,color = 'black', zorder = 20)

#Defining the sigmoid function to predict the probability as output

def model(x):
    return 1/(1+np.exp(-X))

#Using the line formula 
#LinSpace Function is used to evenly space the dots 
#Ravel Function is used to create a single array(Convert 2 arrays into 1 array)
X_test = np.linspace(0, 100, 200)
print(lr.coef_)
print(lr.intercept_)
chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

#Plotting with different colors 
#axhlive stands for axis horizontal live

plt.plot(X_test, chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

# do hit and trial by changing the value of X_test
plt.axvline(x=X_test[180], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(75, 85)
plt.show()
