#requirements
import numpy as np

def euclidean_distance(a, b):
  a = np.array(a) # in case type of a or b is not numpy array
  b = np.array(b)
  return np.sqrt(np.sum((a-b)**2))

def manhattan_distance(a, b):
  a = np.array(a)
  b = np.array(b)
  return np.sum(np.abs(a-b))


def Minkowski_distance(a, b, p=3):
  a = np.array(a)
  b = np.array(b)
  return (np.sum((a-b)**p))**(1/p)

def cossine_similarity(a, b):
  a = np.array(a)
  b = np.array(b)
  magnitude_a = np.sqrt(np.sum(i**2 for i in a))
  magnitude_b = np.sqrt(np.sum(i**2 for i in b))
  return np.dot(a,b)/(magnitude_a*magnitude_b)
