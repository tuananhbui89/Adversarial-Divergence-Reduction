import numpy as np 

def u2s(x):
    """Convert uint8 to [0, 1] float
    """
    return x.astype('float32') / 255

def u2t(x):
    """Convert uint8 to [-1, 1] float
    """
    return x.astype('float32') / 255 * 2 - 1

def s2t(x):
    """Convert [0, 1] float to [-1, 1] float
    """
    return x * 2 - 1