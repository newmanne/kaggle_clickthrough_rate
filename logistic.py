from sklearn.preprocessing import OneHotEncoder
from scipy import io
import numpy as np
import pandas as pd
import csv
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from scipy.sparse import csr_matrix
import pickle
from sklearn.externals import joblib

#id,click,day,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21
# USE_COLS = 'click,day,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21'.split(',')
# DTYPES = {
# 	'click': np.bool_,
# 	'day': np.uint8,
# 	'hour': np.uint8,
# 	'C1': np.uint8,
# 	'banner_pos': np.uint8,
# 	'site_id': np.uint16,
# 	'site_domain': np.uint16,
# 	'site_category': np.uint8,
# 	'app_id': np.uint16,
# 	'app_domain': np.uint8,
# 	'app_category': np.uint8.
# 	'device_id': np.uint16,
# 	'device_ip': np.uint16,
# 	'device_model': np.uint16,
# 	'device_type': np.uint8,
# 	'device_conn_type': np.uint8,
# 	'C14': np.uint16,
# 	'C15': np.uint8,
# 	'C16': np.uint8,
# 	'C17': np.uint16,
# 	'C18': np.uint8,
# 	'C19': np.uint8,
# 	'C20': np.uint8,
# 	'C21': np.uint8
# }
# numpy_dtypes = [np.uint8, np.uint8, np.uint8, np.uint8, np.uint16, np.uint16, np.uint8, np.uint16, np.uint8, np.uint8, np.uint16, np.uint16, np.uint16, np.uint8, np.uint8, np.uint16, np.uint8, np.uint8, np.uint16, np.uint8, np.uint8, np.uint8, np.uint8]
# numpy_dtypes = "u1, u1, u1, u1, u2, u2, u1, u2, u1, u1, u2, u2, u2, u1, u1, u2, u1, u1, u2, u1, u1, u1, u1"
TRAIN_DATA = '../merged_train.csv'
TEST_DATA = '../merged_test.csv'
OUTPUT = '../foo.csv'

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# def iter_loadtxt(filename, delimiter=',', skiprows=0):
#     def iter_func():
#         with open(filename, 'r') as infile:
#             for _ in range(skiprows):
#                 next(infile)
#             for line in infile:
#                 line = line.rstrip().split(delimiter)
#                 for item in line:
#                     yield int(item)
#         iter_loadtxt.rowlength = len(line)

#     data = np.fromiter(iter_func(), dtype=dtype)
#     data = data.reshape((-1, iter_loadtxt.rowlength))
#     return data


	# print "Loading training data"
	# y_train = np.loadtxt(TRAIN_DATA, skiprows=1, usecols=(1,), delimiter=',')
	# np.save('y_train.npy', y_train)
	# print "DONE"

# def first_time_load():
# 	print "Loading training data"
# 	x_test_raw = np.loadtxt(train_path, skiprows=1, usecols=tuple(range(2,25)), dtype='u2', delimiter=',')
# 	np.save('x_test_raw.npy', x_train_raw)
# 	print "DONE"


	# x_train_raw = np.load('x_train_raw.npy')
	# # x_test_raw = np.load('x_test_raw.npy')
	# print "Fitting encoder"
	# n_values =[31,23,6,8,1921,1733,20,1718,118,25,1946,47497,3969,5,5,2163,7,8,453,3,67,159,61]
	# n_values_plus_one = [value + 1 for value in n_values]
	# encoder = OneHotEncoder(n_values_plus_one, dtype='u1')
	# encoder.fit(x_train_raw[0:3])
	# print "Encoder fit. Encoding data"
	# x_train = encoder.transform(x_train_raw)
	# print "done transform. writing file"
	# print x_train.shape
	# np.savez('x_train.npz', data=x_train.data, indices=x_train.indices, indptr=x_train.indptr)
	# x_test_z = np.load('x_test.npz')
	# x_test = csr_matrix((x_test_z['data'], x_test_z['indices'], x_test_z['indptr']), shape=(4577464, 61969))
	# print x_test.dtype
	# print x_test
	# print "Loading testing data"
	# x_test = encoder.transform(test_raw[:, 2:])
	# print "Done encoding data"
	# # y_train = np.load('y_train.npy')
	# y_train = train_raw[:,1]

	# train_raw = pd.read_csv(train_path, usecols=USE_CO S, engine='c', dtypes=DTYPES)
	# x_train_raw = np.zeros((40428967,23))
	# y_train = np.zeros((40428967,1))
	# with open(train_path, 'r') as f:
	#     reader = csv.reader(f)
	#     for i, row in enumerate(reader):
	#     	if i == 0: continue
	#     	if i % 10e6 == 0: print "Training row " + str(i)
	#     	y_train[i-1,0] = row[1]
	#         x_train_raw[i-1,:] = np.array(map(int,row[2:]))
	# np.save('x_train_raw.npy', x_train_raw)
	# np.save('y_train.npy', y_train)
	# train_raw = iter_loadtxt(train_path, skiprows=1)
	
	# train_raw = np.loadtxt(train_path, skiprows=1, delimiter=",")
	# y_train = train_raw[:, 1]

def save_sparse(sparse, path):
	np.savez(path, data=sparse.data, indices=sparse.indices, indptr=sparse.indptr)

def load_sparse(path, n, d):
	x_train_z = np.load('x_train.npz')
	x_train = csr_matrix((x_train_z['data'], x_train_z['indices'], x_train_z['indptr']), shape=(n, d))

def preprocess(train_path, test_path):
	print "Loading training data"
	x_train = load_sparse('x_train.npz', 40428967, 61969)

	
	x_train_z = np.load('x_train.npz')
	x_train = csr_matrix((x_train_z['data'], x_train_z['indices'], x_train_z['indptr']), shape=(40428967, 61969))

	print "Loading testing data"
	x_test_z = np.load('x_test.npz')
	x_test = csr_matrix((x_test_z['data'], x_test_z['indices'], x_test_z['indptr']), shape=(4577464, 61969))

	print "Loading y training"
	y_train = np.load('y_train.npy')

	print "Done load"
	return x_train, y_train, x_test

def train_logistic(x_train, y_train):
	print "Training logistic regression model"
	model = LogisticRegression().fit(x_train, y_train)
	joblib.dump(model, 'logistic.pkl')
	print "Done fitting model"
	return model

def predict(model, x_test):
	print "Making predictions"
	return model.predict_proba(x_test)

if __name__ == '__main__':
	x_train, y_train, x_test = preprocess(TRAIN_DATA, TEST_DATA)
	# model = train_logistic(x_train, y_train)
	# y_hat = predict(model, x_test);
	# print y_hat[:, 1]
	# print "saving output"
	# np.savetxt(OUTPUT, y_hat[:, 1], delimiter=",", fmt='%.14f')