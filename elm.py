import numpy as np
from scipy.linalg import pinv2, inv


class elm():

    def __init__(self, hidden_neurons, activation_function , C=10000,algorithm="no_re"):
        """
        Init method of class elm.

        Initializes all parameters and assigns random values
        to imput weigths and bias
        
        """
        self.hidden_neurons = hidden_neurons
        act = {
            'sigmoid': (lambda x: 1/(1 + np.exp(-x))),
            'tanh': (lambda x: np.tanh(x)),
            'leaky-relu': (lambda x: np.maximum(0.1*x, x)),
            'relu': (lambda x: x*(x > 0)),   
               }
        self.activation_function = act[activation_function]
        self.C = C
        self.algorithm=algorithm

        
        self.b = np.random.normal(0, 1, size=( 1,self.hidden_neurons))


    def calculateH(self, x):

        #Extending w to x shape.
        auxW = self.w

       
        
        auxH = np.dot( x,auxW)+self.b
        self.H = self.activation_function(auxH)


        return self.H

    # compute the output
    def CalculteT(self, H):
        return np.dot(H, self.beta)



    def fit(self,x,y ):
        self.x = x
        self.y = y
        
        self.w = np.random.normal(0, 1, size=(  np.shape(x)[1],self.hidden_neurons))
        self.beta = np.zeros((self.hidden_neurons, np.unique(self.y).shape[0] ))  
        self.H = self.calculateH(self.x)
      
        # no regularization
        if self.algorithm == 'no_re':
            self.beta = np.dot(pinv2(self.H),self.y)
        # faster algorithm 1
        if self.algorithm == 'solution1':
            tmp1 = inv(np.eye(self.H.shape[0])/self.C + np.dot(self.H, self.H.T))
            tmp2 = np.dot( self.H.T,tmp1)
            self.beta = np.dot(tmp2, self.y)
        # faster algorithm 2
        if self.algorithm == 'solution2':
            tmp1 = inv(np.eye(self.H.shape[1])/self.C + np.dot(self.H.T, self.H))
            tmp2 = np.dot(tmp1,self.H.T)
            self.beta = np.dot(tmp2, self.y)
      

        return self
    
    def predict(self, x):
        self.H = self.calculateH(x)
        self.T = self.CalculteT(self.H)
        

        return self.T


      