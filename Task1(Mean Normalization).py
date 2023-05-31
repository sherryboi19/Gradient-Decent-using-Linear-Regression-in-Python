import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("D:\canada_per_capita_income.csv", header=None)
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0]
Y = CSV_Data.iloc[:,1]
copyXSt = X
copyYSt = Y

# Feature Scaling Mean Normaliztion
meanX = np.mean(X) # Mean of X
meanY =  np.mean(Y) # Mean of Y
stdX = np.std(X) # Standard Deviation of X
stdY = np.std(Y) # Standard Deviation of Y


copyXSt = (copyXSt - meanX)/stdX # Normalizing X
copyYSt = (copyYSt - meanY)/stdY # Normalizing Y


# Making rank 2 arrays/
X=np.array(X)
Y=np.array(Y)
copyXSt = np.array(copyXSt)
copyYSt = np.array(copyYSt)
X=X[:,np.newaxis]
Y=Y[:,np.newaxis]
copyXSt = copyXSt[:,np.newaxis]
copyYSt = copyYSt[:,np.newaxis]


#Adding Feature0 or x0
m,col = copyXSt.shape
ones = np.ones((m,1))
copyXSt = np.hstack((ones,copyXSt))
X=np.hstack((ones,X))

#initializing thetas
theta = np.zeros((2,1))

#iterations and alpha
iterations = 8000
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
(h,thetas)=Gradient_Decent_Algo(copyXSt, copyYSt, theta, alpha, iterations, m)

#Predicting Values and Denormalizing Predicted Values
y=np.dot(copyXSt,thetas)
y=y*stdY+meanY

#Making Future Predictions for 2020
tempval=(2020-meanX)/stdX # Normalizing 2020
tempx=np.array([[tempval]]) 
tempone=np.ones((1,1))
tempx=np.hstack((tempone,tempx))
tempy=np.dot(tempx,thetas) # Predicting the Income
tempy=tempy*stdY+meanY # Denormalizing the predicted value

print(tempy) # Printing Predicted Income i.e.  41217.90788934
plt.scatter(X[:,1],Y)
plt.plot(X[:,1],y)
plt.show()
plt.close()