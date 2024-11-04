#!/usr/bin/env python
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

class perceptronX(object):
      
      def __init__(self, learning_rate =  0.001, n_iter = 100):
          
          self.learning_rate =  learning_rate
          self.n_iter =  n_iter
          
          
      def fit(self, X, y):
          
          self.weight =  np.zeros(1+X.shape[1])
          self.errors = list()
          for _ in range(0, self.n_iter):
              error = 0.0
              
              
              for xi, yi in zip(X, y):
                  #1. compute the prediction...
                  ypred =  self.prediction(xi)
                  #2. compute the update..
                  update =  self.learning_rate * (yi - ypred)
                  
                  #3. update weights
                  self.weight[1:] = self.weight[1:] + update * xi
                  self.weight[0]  = self.weight[0] + update
                  
                  error += int(update!=0.0)
                  
              self.errors.append(error)
                  
          return self
          
      def net_input(self, X):
          return np.dot(X, self.weight[1:]) + self.weight[0]
         
      def prediction(self, X):
          return np.where(self.net_input(X) >= 0.0, 1, 0) 

