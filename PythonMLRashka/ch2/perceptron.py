# Perceptron

# z = w'x with w0 = -theta and x0 = 1
# z = net input, w = weight vector, x = input values

# phi(z) = Heaviside step function of theta
# phi(z) = threshold ## Learning Rule:
#  delta w_j = eta(y_true - y_pred) x_j

# w_j = w_j + delta w_j

#  w_j = jth weight, eta = learning rate (0.0, 1.0), 
#  y_true = data, y_pred = output# Perceptron Implementation
import numpy as np

class Perceptron(object):
    '''
    Perceptron classifier
    Parameters:
       eta   : float (learning rate)
       n_iter:   int (passes over the training data set)
    Attributes:
       w_     : 1d-array (weights after fitting)
       errors_:     list (number of misclassification in every epoch)
    '''
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        '''
        Input:
           X: shape = [no_samples, no_features]
              training vector
           y: shape = [n_samples]
              target values
        Returns:
           self: object
        
        '''
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0 ] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            
        return self
    
    def net_input(self, X):
        '''Calculate net input'''
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        '''Returns class label after unit step'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
       
    
    
    
    
    