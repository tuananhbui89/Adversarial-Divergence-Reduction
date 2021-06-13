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

import glob, os, shutil

def backup(source_dir, dest_dir, filetype=".py"):
    if '.' in filetype:
        files = glob.iglob(os.path.join(source_dir, "*{}".format(filetype)))
    else: 
        files = glob.iglob(os.path.join(source_dir, "*.{}".format(filetype)))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, dest_dir)

def list_dir(folder_dir, filetype='.png'):
    if '.' in filetype:
        all_dir = sorted(glob.glob(folder_dir+"*"+filetype), key=os.path.getmtime)
    else:
        all_dir = sorted(glob.glob(folder_dir+"*."+filetype), key=os.path.getmtime)
    return all_dir	

def merge_dict(x, y): 
    merge = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
    return merge