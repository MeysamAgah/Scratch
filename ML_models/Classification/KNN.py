#requirements
import numpy as np
from Distance_metrics.distance_metrics import *

class KNN():
  def __init__(self, k=5, metric = 'euclidean'):
    self.k = k
    self.metric = metric

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):

    # calculating distance according to specified metric
    if self.metric == 'euclidean':
      distances = np.array([euclidean_distance(X,X_train) for X_train in self.X_train])
    elif self.metric == 'manhattan':
      distances = np.array([manhattan_distance(X,X_train) for X_train in self.X_train])
    elif self.metric == 'minkowski':
      distances = np.array([Minkowski_distance(X,X_train) for X_train in self.X_train])
    elif self.metric == 'cossine similarity':
      distances = np.array([cossine_similarity(X,X_train) for X_train in self.X_train])
    else :
      return "invalid metric"

    # getting indexes of nearest labels
    nearest_neighbours_indexes = np.argsort(distances[:self.k])

    # getting labels of nearest neighbours
    nearest_neighbours_labels = [self.y_train[i] for i in nearest_neighbours_indexes]

    #finding frequent label
    def find_frequent_label(NN_labels):
      counter = 0
      num = NN_labels[0]

      for i in NN_labels:
        curr_frequency = NN_labels.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
      return num

    return find_frequent_label(nearest_neighbours_labels)
