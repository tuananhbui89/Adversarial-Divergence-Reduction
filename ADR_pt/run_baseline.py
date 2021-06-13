import subprocess 
import os 

os.chdir('./')
ST = 'python '

stand = dict()
conf = dict()

stand = dict()
stand['ds'] = 'cifar10' 
stand['bs'] = 128
stand['defense'] = 'pgd_train'
stand['model'] = 'resnet18'
stand['epsilon'] = 0.031
stand['trades_beta'] = 1.0

conf['mnist'] = stand.copy()
conf['mnist']['ds'] = 'mnist'
conf['mnist']['model'] = 'cnn'
conf['mnist']['epsilon'] = 0.3 

conf['cifar10'] = stand.copy()
conf['cifar10']['ds'] = 'cifar10'
conf['cifar10']['epsilon'] = 0.031


skip = ['_', '_', '_', '_']

progs = [
	'02a_adversarial_training.py ',
    '02e_evaluate_robustness.py ',
]

for k in list(conf.keys()):
	if k in skip: 
		continue

	for chST in progs: 

		exp = conf[k]
		sub = ' '.join(['--{}={}'.format(t, exp[t]) for t in exp.keys()])
		print(sub)
		subprocess.call([ST + chST + sub], shell=True)