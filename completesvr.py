import numpy as np

from matplotlib import pyplot as plt
import scipy.optimize as so


class SVR():
    def __init__(self, epsilon=0.5,kernel='linear',C=10000,gamma=1,degree=3):
        self.epsilon = epsilon
        self.kernel = {'poly':lambda x,y: np.dot(x, y.T)**degree,
                   'rbf':lambda x,y:np.exp(-gamma*np.abs((x-y)**2)),
                   'linear':lambda x,y: np.dot(x, y.T)}[kernel]
        self.C = C
        
    def _loss(self,alpha):
        
        alpha=alpha.reshape(self.origAshape)
        result=0
        res=0
        for i in range(len(alpha)):
            res=res+np.dot(y[i],alpha[i])
            for j in range(len(alpha)):
                result=result+alpha[i]*alpha[j]*self.kernel(x[i],x[j])
        
        result=-0.5*result-self.epsilon*np.sum(alpha)+res
       
        return -result
    def fit(self, X, y):
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        self.X = X
        self.y = y
        
        self.alpha=np.zeros_like(self.y, dtype=float)
        self.origAshape=np.shape(self.alpha)

        m, n = X.shape
        C=1
        bounds_alpha = so.Bounds(np.zeros(m), np.full(m, C))
        
        result=so.minimize(self._loss,self.alpha,bounds=bounds_alpha)
        
        self.alpha=result.x
        print(self.alpha)
        return self
            
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        self.alpha=self.alpha.reshape(self.origAshape)
        
     
        res=0
        
        for i in range(len(self.alpha)):
            res=res+self.alpha[i]*self.kernel(X[i],X)
            
        return res + self.alpha[0]

