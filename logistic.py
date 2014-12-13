from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

#id,click,day,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21

TRAIN_DATA = '../merged_train_small.csv'
TEST_DATA = '../merged_test_small.csv'
OUTPUT = '../foo.csv'

def preprocess(train_path, test_path):
	print "Loading training data"
	train_raw = np.loadtxt(train_path, skiprows=1, delimiter=",")
	print "Loading testing data"
	test_raw = np.loadtxt(test_path, skiprows=1, delimiter=",")
	print "Combining data for encoder..."
	y_train = train_raw[:, 1]

	print "Fitting encoder"
	n_values =[31,23,6,8,1921,1733,20,1718,118,25,1946,47497,3969,5,5,2163,7,8,453,3,67,159,61]
	n_values_plus_one = [value + 1 for value in n_values]
	encoder = OneHotEncoder(n_values_plus_one)
	encoder.fit(train_raw[:, 2:])
	print "Encoder fit. Encoding data"
	x_train = encoder.transform(train_raw[:, 2:])
	x_test = encoder.transform(test_raw[:, 2:])
	print "Done encoding data"

	return x_train, y_train, x_test

def train_logistic(x_train, y_train):
	model = LogisticRegression().fit(x_train, y_train)
	return model

def predict(model, x_test):
	return model.predict_proba(x_test)

if __name__ == '__main__':
	x_train, y_train, x_test = preprocess(TRAIN_DATA, TEST_DATA)
	model = train_logistic(x_train, y_train)
	y_hat = predict(model, x_test);
	print y_hat[:, 1]
	np.savetxt(OUTPUT, y_hat[:, 1], delimiter=",", fmt='%.14f')