import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta, tempParameter):
    (n,d)=X.shape
    k=len(theta)
    a=np.zeros([k,n])
    tester=np.zeros([k,n])
    for i in range(n):
        for j in range(k):
            x_point=X[i]
            theta_point=theta[j]
            dotin=np.dot(theta_point,x_point)
            tester[j][i]=dotin
    c=np.amax(tester, axis=0)
    for i in range(n):
        for j in range(k):
            x_point=X[i]
            theta_point=theta[j]
            dotin=np.dot(theta_point,x_point)
            a[j][i]=np.exp((dotin/tempParameter)-c[i])
    denom=np.sum(a, axis=0)
    for l in range(n):
        t=denom[l]
        a[:,l]=a[:,l]/denom[l]
    H=a
    return H
    
    
        
            
        
            
    #YOUR CODE HERE
    pass

def computeCostFunction(X, Y, theta, lambdaFactor, tempParameter):
    #regularization cost
    theta_sq=theta*theta
    theta_sq_sum=np.sum(theta_sq)
    reg=theta_sq_sum*lambdaFactor/2
    
    #loss cost
    (n,d)=X.shape
    k=theta.shape[0]
    a=np.zeros([k,n])
    for i in range(n):
        for j in range(k):
            x_point=X[i]
            theta_point=theta[j]
            dotin=np.dot(theta_point,x_point)
            a[j][i]=np.exp(dotin/tempParameter)
    denom=np.sum(a, axis=0)
    for l in range(n):
        t=denom[l]
        a[:,l]=a[:,l]/denom[l]
    alog=np.log(a)
    losscost=0
    for i in range(n):
        for j in range(k):
            if Y[i]==j:
                losscost=losscost+alog[j][i]
    losscost=(-1*losscost)/n
    
    totalcost=losscost+reg
                
    return totalcost
    
    
    
    
    #YOUR CODE HERE
    pass

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter):

    (n,d)=X.shape
    k=theta.shape[0]
    H=computeProbabilities(X, theta, tempParameter)
    
    grad=np.zeros([k,d])
    for j in range(k):
        suma=0
        for i in range(n):
            prob=H[j][i]
            if Y[i]==j:
                suma=suma+(X[i]*(1-prob))
            else:
                suma=suma-((X[i])*prob)
      
        grad[j]=(-suma/(tempParameter*n))+(lambdaFactor*theta[j])
        
    theta=theta-(alpha)*grad
    thetaprime=theta
    return thetaprime
    
        
             
    #YOUR CODE HERE
    pass

def updateY(trainY, testY):
    trainYMod3=np.mod(trainY,3)
    testYmod3=np.mod(testY,3)
    
    return trainYMod3,testYmod3
    #YOUR CODE HERE
    pass

def computeTestErrorMod3(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    assignedLabelsmod3=np.mod(assignedLabels,3)
    return 1 - np.mean(assignedLabelsmod3 == Y)
    
    
    
    
    #YOUR CODE HERE
    pass

def softmaxRegression(X, Y, tempParameter, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor, tempParameter))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter)
    return theta, costFunctionProgression
    
def getClassification(X, theta, tempParameter):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta, tempParameter)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def computeTestError(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    return 1 - np.mean(assignedLabels == Y)
