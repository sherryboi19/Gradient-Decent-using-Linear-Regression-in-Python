import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("D:\CarPrice_Assignment.csv")
CSV_Data = CSV_Data.replace(np.NaN,0)
X1 = CSV_Data.iloc[:,0]
X2 = CSV_Data.iloc[:,1]
X3 = CSV_Data.iloc[:,2]
X4 = CSV_Data.iloc[:,3]
X5 = CSV_Data.iloc[:,4]
X6 = CSV_Data.iloc[:,5]
Y = CSV_Data.iloc[:,6]
copyX1St = X1
copyX2St = X2
copyX3St = X3
copyX4St = X4
copyX5St = X5
copyX6St = X6
copyYSt = Y

# Feature Scaling Mean Normaliztion
meanX1 = np.mean(X1) # Min of X1
meanX2 = np.mean(X2) # Min of X2
meanX3 = np.mean(X3) # Min of X3
meanX4 = np.mean(X4) # Min of X1
meanX5 = np.mean(X5) # Min of X2
meanX6 = np.mean(X6) # Min of X3
meanY =  np.mean(Y) # Min of Y
stdX1 = np.std(X1) # Max of X1
stdX2 = np.std(X2) # Max of X2
stdX3 = np.std(X3) # Max of X3
stdX4 = np.std(X4) # Max of X1
stdX5 = np.std(X5) # Max of X2
stdX6 = np.std(X6) # Max of X3
stdY = np.std(Y) # Max of Y


copyX1St = (copyX1St - meanX1)/stdX1 # Normalizing X1
copyX2St = (copyX2St - meanX2)/stdX2 # Normalizing X2
copyX3St = (copyX3St - meanX3)/stdX3 # Normalizing X3
copyX4St = (copyX4St - meanX4)/stdX4 # Normalizing X1
copyX5St = (copyX5St - meanX5)/stdX5 # Normalizing X2
copyX6St = (copyX6St - meanX6)/stdX6 # Normalizing X3
copyYSt = (copyYSt - meanY)/stdY # Normalizing Y


# Making rank 2 arrays/
X1=np.array(X1)
X2=np.array(X2)
X3=np.array(X3)
X4=np.array(X4)
X5=np.array(X5)
X6=np.array(X6)
Y=np.array(Y)
copyX1St = np.array(copyX1St)
copyX2St = np.array(copyX2St)
copyX3St = np.array(copyX3St)
copyX4St = np.array(copyX4St)
copyX5St = np.array(copyX5St)
copyX6St = np.array(copyX6St)
copyYSt = np.array(copyYSt)
X1=X1[:,np.newaxis]
X2=X2[:,np.newaxis]
X3=X3[:,np.newaxis]
X4=X4[:,np.newaxis]
X5=X5[:,np.newaxis]
X6=X6[:,np.newaxis]
Y=Y[:,np.newaxis]
copyX1St = copyX1St[:,np.newaxis]
copyX2St = copyX2St[:,np.newaxis]
copyX3St = copyX3St[:,np.newaxis]
copyX4St = copyX4St[:,np.newaxis]
copyX5St = copyX5St[:,np.newaxis]
copyX6St = copyX6St[:,np.newaxis]
copyYSt = copyYSt[:,np.newaxis]


#Adding Feature0 or x0 and combining all features
m,col = copyX1St.shape
ones = np.ones((m,1))
copyXSt = np.hstack((ones,copyX1St))
copyXSt = np.hstack((copyXSt,copyX2St))
copyXSt = np.hstack((copyXSt,copyX3St))
copyXSt = np.hstack((copyXSt,copyX4St))
copyXSt = np.hstack((copyXSt,copyX5St))
copyXSt = np.hstack((copyXSt,copyX6St))
X=np.hstack((ones,X1))
X=np.hstack((X,X2))
X=np.hstack((X,X3))
X=np.hstack((X,X4))
X=np.hstack((X,X5))
X=np.hstack((X,X6))
#initializing thetas
theta = np.zeros((7,1))

#iterations and alpha
iterations = 30000
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
(h,thetas)=Gradient_Decent_Algo(copyXSt[0:154,:], copyYSt[0:154,:], theta, alpha, iterations, m)

#Predicting Values and Denormalizing Predicted Values
y=np.dot(copyXSt,thetas)
y=y*stdY+meanY

#Making Future Predictions for 2020
tempx=copyXSt[154:,:]
tempy=np.dot(tempx,thetas) # Predicting the Income
tempy=tempy*stdY+meanY # Denormalizing the predicted value

err = tempy-Y[154:,:]


print(tempy)
print(f"The Error :\n {err}")
plt.plot(h)