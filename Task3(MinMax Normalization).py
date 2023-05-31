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
minX1 = np.min(X1) # Min of X1
minX2 = np.min(X2) # Min of X2
minX3 = np.min(X3) # Min of X3
minX4 = np.min(X4) # Min of X4
minX5 = np.min(X5) # Min of X5
minX6 = np.min(X6) # Min of X6
minY =  np.min(Y) # Min of Y
maxX1 = np.max(X1) # Max of X1
maxX2 = np.max(X2) # Max of X2
maxX3 = np.max(X3) # Max of X3
maxX4 = np.max(X4) # Max of X4
maxX5 = np.max(X5) # Max of X5
maxX6 = np.max(X6) # Max of X6
maxY = np.max(Y) # Max of Y

t1=maxX1-minX1
t2=maxX2-minX2
t3=maxX3-minX3
t4=maxX1-minX4
t5=maxX2-minX5
t6=maxX3-minX6
tY=maxY-minY
copyX1St = (copyX1St - minX1)/t1 # Normalizing X1
copyX2St = (copyX2St - minX2)/t2 # Normalizing X2
copyX3St = (copyX3St - minX3)/t3 # Normalizing X3
copyX4St = (copyX4St - minX4)/t4 # Normalizing X4
copyX5St = (copyX4St - minX5)/t5 # Normalizing X5
copyX6St = (copyX4St - minX5)/t6 # Normalizing X6
copyYSt = (copyYSt - minY)/tY # Normalizing Y


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
y=y*tY+minY

#Making Future Predictions for 2020
tempx=copyXSt[154:,:]
tempy=np.dot(tempx,thetas) # Predicting the values
tempy=tempy*tY+minY # Denormalizing the predicted value

err = tempy-Y[154:,:]


print(tempy)
print(f"The Error :\n {err}")
plt.plot(h)