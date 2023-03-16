from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
from random import random





class mlp():

    def __init__(self, hidden_layer_sizes=(1,),activation_function="sigmoid",max_iter=2000 ,learning_rate=0.1,random_state=1):
        """
        Init method of class elm.

        Initializes all parameters and assigns random values
        to imput weigths and bias
        
        """
        np.random.seed(random_state)
        self.hidden_layers=len(hidden_layer_sizes)+1
        
        self.hidden_neurons=hidden_layer_sizes
       
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        act = {
            'sigmoid': (lambda x: 1/(1 + np.exp(-x))),
            'tanh': (lambda x: np.tanh(x)),
            'leaky-relu': (lambda x: np.maximum(0.1*x, x)),
            'relu': (lambda x: x*(x > 0)),   
               }

        der = {
            'sigmoid': (lambda x: np.exp(-x)/np.power((1+np.exp(-x)),2)),
            'tanh': (lambda x: 1-np.tanh(x)**2),
            'leaky-relu': (lambda x: np.where(x>0, 1, 0.1) ),
            'relu': (lambda x: 1 * (x>0))
               }     
        

        self.act = act[activation_function] 
        self.der = der[activation_function]   
             

    def _predict(self, x):


        """
        calculates weight matriz for every layer
        
        """
        

        self.H_list=[]
        
        for i in range(self.hidden_layers):
           
            auxW = self.wb[0][i]
            auxB = self.wb[1][i]
            auxH =  x @ auxW + auxB
            H=self.act(auxH)
            
            x=H
            
            self.H_list.append(H)
        
       
        return H


    def predict(self, x):

        x=(x+(np.abs(np.min(x))))/((np.abs(np.max(x))+np.abs(np.min(x))))
        H=self._predict(x)
        H=H*self.scale_y+self.min_y
     
        return H
    def _error(self):
        return (np.mean(np.power(self.H_list[-1] - self.y,2)))/2
    
    def _backpropagate(self):

        #First Layer
        i=1
        delta=( self.H_list[-i]-self.y)*self.der(self.H_list[-i])
        output_gradients=self.H_list[-(i+1)].T @ delta

        self.wb[0][-1]=self.wb[0][-i]-output_gradients*self.learning_rate
        self.wb[1][-1]=self.wb[1][-i]-np.sum(delta, axis=0, keepdims=True) * self.learning_rate

        #Middle Layers

        for i in (range(2,self.hidden_layers)):

            
            delta=(delta @ self.wb[0][-(i-1)].T)*self.der(self.H_list[-i])

            output_gradients=self.H_list[-(i+1)].T @ delta          


            self.wb[0][-i]=self.wb[0][-i]-output_gradients*self.learning_rate
            self.wb[1][-i]=self.wb[1][-i]-np.sum(delta, axis=0, keepdims=True) * self.learning_rate

        #Last Layer

        i=self.hidden_layers+1
        
        delta=(delta @ self.wb[0][1].T) * self.der(self.H_list[0])

        output_gradients=self.x.T @ delta

          
        self.wb[0][0]=self.wb[0][0]-output_gradients*self.learning_rate
        self.wb[1][0]=self.wb[1][0]-np.sum(delta, axis=0, keepdims=True) * self.learning_rate

    def _backpropagate2(self):
        #update output

        for i in (range(1,self.hidden_layers+1)):

            
            if i==1:
                delta=( self.H_list[-i]-self.y)*self.der(self.H_list[-i])
            elif i>1 :
                delta=(delta @ self.wb[0][-(i-1)].T)    
            
            delta=delta*self.der(self.H_list[-i])

            if i<self.hidden_layers:
                output_gradients=self.H_list[-(i+1)].T @ delta          
            else:
                output_gradients=self.x.T @ delta
            
            self.wb[0][-i]=self.wb[0][-i]-output_gradients*self.learning_rate
            self.wb[1][-i]=self.wb[1][-i]-np.sum(delta, axis=0, keepdims=True) * self.learning_rate
           
       
    def fit(self,x,y):

      
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        self.x = (x+(np.abs(np.min(x))))/((np.abs(np.max(x))+np.abs(np.min(x))))
        self.scale_x=np.max(x)-np.min(x)
        
        self.scale_y=np.max(y)-np.min(y)
        self.min_y=np.min(y)
        
        self.y = (y+np.abs(np.min(y)))/(np.abs(np.max(y))+np.abs(np.min(y)))


   
        w = []
        b = []
        #initialization of weigth and bias matrices
        self.errors=[]
        
        row_len = np.shape(x)[1]
        col_len = self.hidden_neurons[0]

        for i in range(0,self.hidden_layers):

          
            if i<self.hidden_layers-1:
                col_len = self.hidden_neurons[i]
            
            else:
                col_len = 1
            
            auxW=(np.random.uniform(size= (row_len,col_len)))
            auxB=(np.random.uniform(size= (1,col_len)))
            
            w.append(auxW)
            b.append(auxB)
            #update next shapes

            
            row_len = col_len
            

        self.wb=[w,b]

        
                
        for i in range(self.max_iter):
            self._predict(self.x)
            
            self.errors.append(self._error())
         
            #backward propagation

            self._backpropagate()
            

        return self

    
    
