import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("D:\hiring.csv")
CSV_Data = CSV_Data.replace(np.NaN,0)
X1 = CSV_Data.iloc[:,0]
X2 = CSV_Data.iloc[:,1]
X3 = CSV_Data.iloc[:,2]
Y = CSV_Data.iloc[:,3]
copyX1St = X1
copyX2St = X2
copyX3St = X3
copyYSt = Y

# Feature Scaling Mean Normaliztion
meanX1 = np.mean(X1) # Mean of X1
meanX2 = np.mean(X2) # Mean of X2
meanX3 = np.mean(X3) # Mean of X3
meanY =  np.mean(Y) # Mean of Y
stdX1 = np.std(X1) # Standard Deviation of X1
stdX2 = np.std(X2) # Standard Deviation of X2
stdX3 = np.std(X3) # Standard Deviation of X3
stdY = np.std(Y) # Standard Deviation of Y


copyX1St = (copyX1St - meanX1)/stdX1 # Normalizing X1
copyX2St = (copyX2St - meanX2)/stdX2 # Normalizing X2
copyX3St = (copyX3St - meanX3)/stdX3 # Normalizing X3
copyYSt = (copyYSt - meanY)/stdY # Normalizing Y


# Making rank 2 arrays/
X1=np.array(X1)
X2=np.array(X2)
X3=np.array(X3)
Y=np.array(Y)
copyX1St = np.array(copyX1St)
copyX2St = np.array(copyX2St)
copyX3St = np.array(copyX3St)
copyYSt = np.array(copyYSt)
X1=X1[:,np.newaxis]
X2=X2[:,np.newaxis]
X3=X3[:,np.newaxis]
Y=Y[:,np.newaxis]
copyX1St = copyX1St[:,np.newaxis]
copyX2St = copyX2St[:,np.newaxis]
copyX3St = copyX3St[:,np.newaxis]
copyYSt = copyYSt[:,np.newaxis]


#Adding Feature0 or x0 and combining all features
m,col = copyX1St.shape
ones = np.ones((m,1))
copyXSt = np.hstack((ones,copyX1St))
copyXSt = np.hstack((copyXSt,copyX2St))
copyXSt = np.hstack((copyXSt,copyX3St))
X=np.hstack((ones,X1))
X=np.hstack((X,X2))
X=np.hstack((X,X3))

#initializing thetas
theta = np.zeros((4,1))

#iterations and alpha
iterations = 1000
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
tempx1=(12-meanX1)/stdX1 # Normalizing 2.12year experience
tempx2=(10-meanX2)/stdX2 # Normalizing 9,10test score
tempx3=(10-meanX3)/stdX3 # Normalizing 6,10interview score
tempx=np.array([[tempx1,tempx2,tempx3]]) 
tempone=np.ones((1,1))
tempx=np.hstack((tempone,tempx))
tempy=np.dot(tempx,thetas) # Predicting the Income
tempy=tempy*stdY+meanY # Denormalizing the predicted value

print(tempy) # Printing Predicted Income i.e.  Part1/ 485.45013565   Part2/957.55814812
plt.plot(h)