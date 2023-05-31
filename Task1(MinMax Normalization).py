import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("D:\canada_per_capita_income.csv", header=None)
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0]
Y = CSV_Data.iloc[:,1]
copyXNm = X
copyYNm = Y

# Feature Scaling MinMax Normaliztion
minX = np.min(X) # Min of X
minY =  np.min(Y) # Minn of Y
maxX = np.max(X) # Max of X
maxY = np.max(Y) # Max of Y

tX=maxX-minX
tY=maxY-minY
copyXNm = (copyXNm-minX)/tX # MinMax Normalization
copyYNm = (copyYNm-minY)/tY # MinMax Normalization


# Making rank 2 arrays/
X=np.array(X)
Y=np.array(Y)
copyXNm = np.array(copyXNm)
copyYNm = np.array(copyYNm)
X=X[:,np.newaxis]
Y=Y[:,np.newaxis]
copyXNm = copyXNm[:,np.newaxis]
copyYNm = copyYNm[:,np.newaxis]


#Adding Feature0 or x0
m,col = copyXNm.shape
ones = np.ones((m,1))
copyXNm = np.hstack((ones,copyXNm))
X=np.hstack((ones,X))

#initializing thetas
theta = np.zeros((2,1))

#iterations and alpha
iterations = 4000
alpha = 0.01



# Defining Cost function

def Get_cost_J(X,Y,Theta):
    Pridictions = np.dot(X,Theta)
    Error = Pridictions-Y
    SqrError = np.power(Error,2)
    SumSqrError = np.sum(SqrError)
    J  = (1/2*m)*SumSqrError # Where m is tototal number of rows
    return J

#Defining Gradient Decent Algorithm



def Gradient_Decent_Algo(X,Y,Theta,alpha,itrations,m):
    histroy = np.zeros((itrations,1))
    for i in range(itrations):
        temp =(np.dot(X,Theta))-Y
        temp = (np.dot(X.T,temp))*alpha/m
        Theta = Theta - temp
             
        histroy[i] = Get_cost_J(X, Y, Theta)
       
    return (histroy,Theta)

#Calling Function and Storing History and Thetas
(h,thetas)=Gradient_Decent_Algo(copyXNm, copyYNm, theta, alpha, iterations, m)

#Predicting Values and Denormalizing Predicted Values
y=np.dot(copyXNm,thetas)
y=y*tY+minY

#Making Future Predictions for 2020
tempval=(2020-minX)/tX # Normalizing 2020
tempx=np.array([[tempval]]) 
tempone=np.ones((1,1))
tempx=np.hstack((tempone,tempx))
tempy=np.dot(tempx,thetas) # Predicting the Income
tempy=tempy*tY+minY # Denormalizing the predicted value

print(tempy) # Printing Predicted Income 40188.93502514
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],y)
plt.show()
plt.close()