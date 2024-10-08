#Note
## You need to scale data before fitting the model

#requiremets
import numpy as np

class LinearRegression():
  def __init__(self, lr=0.01, n_iters=1000, L2=0, L1=0):
    self.lr = lr
    self.n_iters = n_iters
    self.L2 = L2
    self.L1 = L1

  def fit(self, X, y):
    num_samples, num_features = X.shape
    self.weights = np.zeros(num_features)
    self.bias = 0

    for i in range(self.n_iters):
      y_pred = np.dot(X, self.weights) + self.bias
      dw = (2/num_samples) * (np.dot(X.T, (y_pred - y))+ (self.L2 * self.weights) + (self.L1 * self.weights))
      db = (2/num_samples) * np.sum(y_pred - y)

      self.weights -= self.lr * dw
      self.bias -= self.lr * db


  def predict(self, X):
    y_pred = np.dot(X, self.weights) + self.bias
    return y_pred
