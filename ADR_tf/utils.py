import os 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib as mpl

from sklearn.manifold import TSNE
import seaborn as sns

import glob, os, shutil

def backup(source_dir, dest_dir):
	files = glob.iglob(os.path.join(source_dir, "*.py"))
	for file in files:
		if os.path.isfile(file):
			shutil.copy2(file, dest_dir)


def cvtord(ord): 
	return int(ord) if ord != 'inf' else np.inf

def save_model(saver, sess, model_dir, global_step):
    path = saver.save(sess, os.path.join(model_dir, 'model'),
                      global_step=global_step)
    print("Saving model to {}".format(path))

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def convert2onehot(x): 
    b = np.zeros((x.size, x.max()+1))
    b[np.arange(x.size),x] = 1
    return b 

def onehot2score(onehot): 
    n,d = np.shape(onehot)
    x = np.zeros(shape=[n,d])
    for i in range(d): 
        x[:,i] = onehot[:,i]*(i+1)
    s = np.sum(x, axis=1)
    return s 

def onehot2entropy(onehot): 
	return np.sum(-1.*onehot * np.log(onehot+1e-10), axis=1)

def onehot2class(onehot): 
    return np.argmax(onehot, axis=1)

def est_entropy(x): 
	p = np.exp(x)/sum(np.exp(x))
	e = sum(p*np.log(p)) 
	return -e 

class LogData(object):
	"""docstring for LogData"""
	def __init__(self):
		super(LogData, self).__init__()
		self.data = dict()
	
	def log(self, key, value): 
		if key not in self.data.keys(): 
			self.data[key] = []

		self.data[key].append(value)

	def mean(self, key): 
		return np.mean(self.data[key])

	def plot(self, savepath, key, x=None): 
		plt.figure()
		if x is None:
			if 'iter' in self.data.keys():
				plt.plot(self.data['iter'], self.data[key])
			else:
				plt.plot(np.arange(len(self.data[key])), self.data[key])
		else: 
			plt.plot(x, self.data[key])
		plt.grid(True)
		plt.xlabel('Iteration')
		plt.ylabel('Value of {}'.format(key))
		plt.savefig(savepath, dpi=300)
		plt.close()

	def plot2(self, savepath, key, x=None): 
		plt.figure()
		if x is None:
			if 'iter' in self.data.keys():
				plt.plot(self.data['iter'], self.data[key], color='g', marker='+', label='train')
				plt.plot(self.data['iter'], self.data[key+'v2'], color='r', marker='d', label='eval')
			else:
				plt.plot(np.arange(len(self.data[key])), self.data[key], color='g', marker='+', label='train')
				plt.plot(np.arange(len(self.data[key+'v2'])), self.data[key+'v2'], color='r', marker='d', label='eval')
		else: 
			plt.plot(x, self.data[key], color='g', marker='+', label='train')
			plt.plot(x, self.data[key+'v2'], color='r', marker='d', label='eval')
		plt.grid(True)
		plt.xlabel('Iteration')
		plt.ylabel('Value of {}'.format(key))
		plt.legend()
		plt.savefig(savepath, dpi=300)
		plt.close()

	def lognplot(self, key, value, savepath): 
		self.log(key=key, value=value)
		self.plot(savepath=savepath, key=key)

	def lognplot2(self, key, value, value2, savepath): 
		self.log(key=key, value=value)
		self.log(key=key+'v2', value=value2)
		self.plot2(savepath=savepath, key=key)


		