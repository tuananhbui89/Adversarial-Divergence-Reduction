import os 
import numpy as np 

from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import matplotlib.pyplot as plt 

n_train = 10000
n_test = 1000
rs_train = 2019
rs_test = 2020

def gen_blobs(n_samples, rs): 
	centers = [(0, 0), (0, 5), (5, 0), (5, 5)] 
	cluster_std = 0.5
	prob = [0.25, 0.25, 0.5]
	data, labels = datasets.make_blobs(n_samples=n_samples, random_state=rs, centers=centers, cluster_std=cluster_std)
	labels = np.where(labels>2,2,labels)	
	return (data, labels), 3

def gen_circles(n_samples, rs): 
	prob = [0.5, 0.5]
	d1, l1 = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05,random_state=rs)
	_d2, _l2 = datasets.make_circles(n_samples=n_samples, factor=0.8,noise=0.95,random_state=rs+1)
	d2 = _d2[_l2==0]
	l2 = 3*np.ones((np.shape(d2)[0],))
	data = np.concatenate([d1, d2], axis=0)
	labels = np.concatenate([l1, l2], axis=0)
	return (data, labels), 3

def gen_moons(n_samples, rs): 
	prob = [0.5, 0.5]
	d1, l1 = datasets.make_moons(n_samples=n_samples, noise=.05,random_state=rs)
	_d2, _l2 = datasets.make_moons(n_samples=n_samples, noise=.1,random_state=rs+1)
	_d2 = _d2 + 2*np.max(np.abs(d1))
	d2 = _d2[_l2 == 0]
	l2 = 3*np.ones((np.shape(d2)[0],))
	data = np.concatenate([d1, d2], axis=0)
	labels = np.concatenate([l1, l2], axis=0)
	return (data, labels), 3

def gen_data(ds, n_samples, rs): 
	if ds == 'blobs': 
		(data, labels), nbc = gen_blobs(n_samples, rs)
	elif ds == 'circles': 
		(data, labels), nbc = gen_circles(n_samples, rs)
	elif ds == 'moons': 
		(data, labels), nbc = gen_moons(n_samples, rs)
	return (data, labels), nbc

def gen_n_save(savepath, ds): 
	(trainx, trainy), nbc = gen_data(ds, n_train, rs_train)
	(testx, testy), _ = gen_data(ds, n_test, rs_test)
	data = dict()
	data['trainx'] = trainx
	data['trainy'] = trainy
	data['testx'] = testx
	data['testy'] = testy
	data['cls'] = nbc
	np.save(savepath, data)

def main(): 
	for ds in ['blobs', 'circles', 'moons']:
		gen_n_save('toy2d_{}.npy'.format(ds), ds)

def test(): 
	for ds in ['blobs', 'circles', 'moons']: 
		data = np.load('toy2d_{}.npy'.format(ds), allow_pickle=True).item()
		plt.scatter(data['testx'][0], data['testx'][1])
		plt.show()
		for k in data.keys(): 
			print('key {}, shape {}, max {}, min {}'.format(k, np.shape(data[k]), np.max(data[k]), np.min(data[k])))

if __name__ == '__main__': 
	main()
	test()