#requirements
import numpy as np

def MAE(y_true, y_pred):
  return np.mean(np.abs(y_true - y_pred))

def MSE(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)

def RMSE(y_true, y_pred):
  return np.sqrt(MSE(y_true, y_pred))

def RMSLE(y_true, y_pred):
  return np.log(RMSE(y_true, y_pred))

def R2(y_true, y_pred):
  RSS = np.sum((y_true - y_pred)**2)
  TSS = np.sum((y_true - np.mean(y_true))**2)
  return 1 - (RSS/TSS)

def R2_adjusted(y_true, y_pred, n, k):
  r2 = R2(y_true, y_pred)
  return 1 - ((1-r2)*(n-1)/(n-k-1))
