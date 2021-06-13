import os 
import numpy as np 
import shutil

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copyfile(src, dst):
    path = os.path.dirname(dst)
    mkdir_p(path)
    shutil.copyfile(src, dst)

def chdir_p(path='/content/drive/My Drive/Workspace/OT/myOT/'): 
    os.chdir(path)
    WP = os.path.dirname(os.path.realpath('__file__')) +'/'
    print('CHANGING WORKING PATH: ', WP)

def writelog(data=None, logfile=None, printlog=True):
    fid = open(logfile,'a')
    fid.write('%s\n'%(data))
    fid.flush()
    fid.close()
    if printlog: 
        print(data)

def dict2str(d): 
    # assert(type(d)==dict)
    res = ''
    for k in d.keys(): 
        v = d[k]
        res = res + '{}:{},'.format(k,v)
    return res 

def list2str(l): 
    # assert(type(l)==list)
    res = ''
    for i in l: 
        res = res + ' {}'.format(i)
    return res 

def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def delete_existing(path, overwrite=True):
    """Delete directory if it exists

    Used for automatically rewrites existing log directories
    """
    if not overwrite:
        assert not os.path.exists(path), "Cannot overwrite {:s}".format(path)
    else:
        if os.path.exists(path):
            shutil.rmtree(path)