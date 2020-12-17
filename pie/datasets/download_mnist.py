import os
import numpy as np
import subprocess
from scipy.io import loadmat, savemat
from skimage.transform import resize

def mnist_resize(x):
    H, W, C = 32, 32, 3
    x = x.reshape(-1, 28, 28)
    resized_x = np.empty((len(x), H, W), dtype='float32')
    for i, img in enumerate(x):
        # resize returns [0, 1]
        resized_x[i] = resize(img, (H, W), mode='reflect')

    # Retile to make RGB
    resized_x = resized_x.reshape(-1, H, W, 1)
    resized_x = np.tile(resized_x, (1, 1, 1, C))
    return resized_x

def main():
    if os.path.exists('mnist.npz'):
        print("Using existing mnist.npz")

    else:
        print("Opening subprocess to download data from URL")
        subprocess.check_output('wget https://s3.amazonaws.com/img-datasets/mnist.npz',shell=True)

    if os.path.exists('mnist32_train.mat') and os.path.exists('mnist32_test.mat'):
        print("Using existing mnist32_train.mat and mnist32_test.mat")

    else:
        print("Resizing mnist.npz to (32, 32, 3)")
        data = np.load('mnist.npz')
        trainx = data['x_train']
        trainy = data['y_train']
        savemat('mnist28_train.mat', {'X': trainx/255., 'y': trainy})
        trainx = mnist_resize(trainx)
        savemat('mnist32_train.mat', {'X': trainx, 'y': trainy})

        testx = data['x_test']
        testy = data['y_test']
        savemat('mnist28_test.mat', {'X': testx/255., 'y': testy})
        testx = mnist_resize(testx)
        savemat('mnist32_test.mat', {'X': testx, 'y': testy})

    print("Loading mnist28_train.mat for sanity check")
    data = loadmat('mnist28_train.mat')
    print(data['X'].shape, data['X'].min() ,data['X'].max()) # (60000, 28, 28) 0.0 1.0
    print(data['y'].shape, data['y'].min() ,data['y'].max()) # (1, 60000) 0 9

    print("Loading mnist28_test.mat for sanity check")
    data = loadmat('mnist28_test.mat')
    print(data['X'].shape, data['X'].min() ,data['X'].max()) # (10000, 28, 28) 0.0 1.0
    print(data['y'].shape, data['y'].min() ,data['y'].max()) # (1, 10000) 0 9

    print("Loading mnist32_train.mat for sanity check")
    data = loadmat('mnist32_train.mat')
    print(data['X'].shape, data['X'].min() ,data['X'].max()) # (60000, 32, 32, 3) 0.0 1.0
    print(data['y'].shape, data['y'].min() ,data['y'].max()) # (1, 60000) 0 9

    print("Loading mnist32_test.mat for sanity check")
    data = loadmat('mnist32_test.mat')
    print(data['X'].shape, data['X'].min() ,data['X'].max()) # (10000, 32, 32, 3) 0.0 1.0
    print(data['y'].shape, data['y'].min() ,data['y'].max()) # (1, 10000) 0 9


if __name__ == '__main__':
    main()
