import subprocess
import os
from scipy.io import loadmat 

def main():
    if os.path.exists('svhn_test_32x32.mat') and os.path.exists('svhn_train_32x32.mat'):
        print("Using existing data")

    else:
        print("Opening subprocess to download data from URL")
        subprocess.check_output(
            '''
            wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
            wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
            wget http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
            ''',
            shell=True)

        os.rename('test_32x32.mat', 'svhn_test_32x32.mat')
        os.rename('train_32x32.mat', 'svhn_train_32x32.mat')
        os.rename('extra_32x32.mat', 'svhn_extra_32x32.mat')

    print("Loading svhn_train_32x32.mat for sanity check")
    data = loadmat('svhn_train_32x32.mat')
    print(data['X'].shape, data['X'].min() ,data['X'].max()) # (32, 32, 3, 73257) 0 255
    print(data['y'].shape, data['y'].min() ,data['y'].max()) # (73257, 1) 1 10

    print("Loading svhn_test_32x32.mat for sanity check")
    data = loadmat('svhn_test_32x32.mat')
    print(data['X'].shape, data['X'].min() ,data['X'].max()) # (32, 32, 3, 26032) 0 255
    print(data['y'].shape, data['y'].min() ,data['y'].max()) # (26032, 1) 1 10


if __name__ == '__main__':
    main()
