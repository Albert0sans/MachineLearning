import numpy as np

from matplotlib import pyplot as plt
import scipy.optimize as so


class SVR():
    def __init__(self, epsilon=0.5,kernel='linear',C=1,gamma=1,degree=4):
        self.epsilon = epsilon
        self.kernel = {'poly':lambda x,y: np.dot(x, y.T)**degree,
                   'rbf':lambda x,y:np.exp(-gamma*np.sum((y-x[:,np.newaxis])**2,axis=-1)),
                   'linear':lambda x,y: np.dot(x, y.T)}[kernel]
        self.C = C
        
        
    def _loss(self,params):
        W,b=params
        W=W.reshape(self.origWshape)
        b=b.reshape(self.origbshape)
        
        y_pred=np.matmul(self.X,W)-b
        loss=np.linalg.norm(W)/2 + np.mean(np.maximum(0., np.abs(y_pred- self.y) - self.epsilon))
        
        return loss
    def fit(self, X, y, epochs=100, learning_rate=0.1):
        
        
        feature_len = X.shape[-1] if len(X.shape) > 1 else 1
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        self.X = X
        self.y = y
        

        self.W=(np.random.uniform(size= (feature_len,1)))
        self.origWshape=np.shape(self.W)
        self.b=(np.random.uniform(size= (1,1)))
        self.origbshape=np.shape(self.b)

        m, n = X.shape
        C=1
        
        
        
        result=so.minimize(self._loss,[self.W,self.b])
        
        self.W,self.b=result.x

        
            
        return self
            
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        self.W=self.W.reshape(self.origWshape)
        self.b=self.b.reshape(self.origbshape)
        print("w ",self.W,"b ",self.b)
        
        #return np.matmul(self.X, self.W) - self.b
        return np.matmul(self.X,self.W)-self.b

