import os 
import numpy as np 
from pie.utils.utils_data import u2t, s2t
from scipy.io import loadmat 

class Data(object):
	"""docstring for Data"""
	def __init__(self, images, labels=None, cast=False, seed=2020):
		"""
		Data object constructs mini batches to be fed during training 
		Args: 
			images: Input images [N, H, W, C]
			labels: One hot vectors [N, K]
			cast: bool, convert uint8 [0,255] to [-1., 1.] float
			seed: random seed
		"""
		super(Data, self).__init__()
		self.cast = cast
		self.images = self.preprocess(images)
		self.labels = labels
		self.seed = seed if seed is not None else 2020 
		self.num_samples = np.shape(self.images)[0]
		np.random.seed(self.seed)

	def preprocess(self, x): 
		if self.cast: 
			assert(np.max(x)==255)
			assert(np.min(x)==0)
			return u2t(x)
		else: 
			return x 

	def filtered_batch(self, bs, target=None): 
		if target is None: 
			return self.next_batch(bs)
		else: 
			assert(type(target) == int)
			valid = np.arange(self.num_samples)[self.labels == target]
			idx = np.random.choice(valid, bs, replace=False)
			x = self.images[idx]
			y = self.labels[idx]
			return x, y

	def next_batch(self, bs): 
		idx = np.random.choice(self.num_samples, bs, replace=False)
		x = self.images[idx]
		y = self.labels[idx]
		return x, y 

	def get_one(self): 
		return self.next_batch(bs=1)

class Mnist(object):
	"""docstring for Mnist"""
	def __init__(self, datadir):
		super(Mnist, self).__init__()
		print("Loading MNIST")
		train = loadmat(os.path.join(datadir, 'mnist28_train.mat'))
		test = loadmat(os.path.join(datadir, 'mnist28_test.mat'))

		assert(np.min(train['X'])==0.)
		assert(np.max(train['X'])==1.)
		trainx = train['X']
		trainy = train['y'].reshape(-1)
		trainy = np.eye(10)[trainy].astype('float32')

		testx = test['X']
		testy = test['y'].reshape(-1)
		testy = np.eye(10)[testy].astype('float32')

		trainx = trainx.reshape(-1, 28, 28, 1).astype('float32')
		testx = testx.reshape(-1, 28, 28, 1).astype('float32')

		self.train = Data(images=trainx, labels=trainy)
		self.test = Data(images=testx, labels=testy)

class Mnist32(object):
	"""docstring for Mnist"""
	def __init__(self, datadir):
		super(Mnist, self).__init__()
		print("Loading MNIST")
		train = loadmat(os.path.join(datadir, 'mnist32_train.mat'))
		test = loadmat(os.path.join(datadir, 'mnist32_test.mat'))

		assert(np.min(train['X'])==0.)
		assert(np.max(train['X'])==1.)
		trainx = train['X']
		trainy = train['y'].reshape(-1)
		trainy = np.eye(10)[trainy].astype('float32')

		testx = test['X']
		testy = test['y'].reshape(-1)
		testy = np.eye(10)[testy].astype('float32')

		trainx = trainx.reshape(-1, 32, 32, 3).astype('float32')
		testx = testx.reshape(-1, 32, 32, 3).astype('float32')

		self.train = Data(images=trainx, labels=trainy)
		self.test = Data(images=testx, labels=testy)

class Svhn(object):
    def __init__(self, datadir):
        """SVHN domain train/test data

        train - (str) flag for using 'train' or 'extra' data
        """
        print("Loading SVHN")
        train = loadmat(os.path.join(datadir, 'svhn_train_32x32.mat'))
        test = loadmat(os.path.join(datadir, 'svhn_test_32x32.mat'))

        # Change format
        trainx, trainy = self.change_format(train)
        testx, testy = self.change_format(test)

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

    @staticmethod
    def change_format(mat):
        """Convert X: (HWCN) -> (NHWC) and Y: [1,...,10] -> one-hot
        """
        x = mat['X'].transpose((3, 0, 1, 2))
        y = mat['y'].reshape(-1)
        y[y == 10] = 0
        y = np.eye(10)[y]
        return x, y

class Cifar10(object):
    def __init__(self, datadir):
        """CIFAR-10 modified domain train/test data

        Modification: one of the classes was removed to match STL
        """
        print("Loading CIFAR")
        train = loadmat(os.path.join(datadir, 'cifar10_train.mat'))
        test = loadmat(os.path.join(datadir, 'cifar10_test.mat'))

        # Get data
        trainx, trainy = train['X'], train['y']
        testx, testy = test['X'], test['y']

        # Convert to one-hot 
        # NOTICE HERE: ORIGINAL CODE np.eye(9) ---> Modify to np.eye(10)

        trainy = np.eye(10)[trainy.reshape(-1)]
        testy = np.eye(10)[testy.reshape(-1)]

        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)

class Toy2D(object):
	"""docstring for Toy2D"""
	def __init__(self, datadir, ds='circles'):
		print("Loading Toy2D: {}".format(ds))
		data = np.load(os.path.join(datadir, 'toy2d_{}.npy'.format(ds)), allow_pickle=True).item()

		# Get data
		trainx, trainy = data['trainx'], data['trainy']
		testx, testy = data['testx'], data['testy']

		trainy = np.eye(data['cls'])[trainy.reshape(-1)]
		testy = np.eye(data['cls'])[testy.reshape(-1)]

		self.train = Data(trainx, trainy)
		self.test = Data(testx, testy)     


def get_data(ds, datadir): 
	if ds == 'mnist': 
		return Mnist(datadir)
	elif ds == 'mnist32': 
		return Mnist32(datadir)
	elif ds == 'svhn': 
		return Svhn(datadir)
	elif ds == 'cifar10': 
		return Cifar10(datadir)
	elif ds == 'blobs': 
		return Toy2D(datadir, 'blobs')
	elif ds == 'circles': 
		return Toy2D(datadir, 'circles')
	elif ds == 'moons': 
		return Toy2D(datadir, 'moons')
	else: 
		raise Exception("dataset {} not recognized".format(ds))		


