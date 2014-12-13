from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def knn_train(x_train, y_train):
	neigh = KNeighborsRegressor(n_neighbors=2)
	neigh.fit(X, y) 
	return neigh

def knn_test(model, y_train):
	print(model.predict([[1.5]]))