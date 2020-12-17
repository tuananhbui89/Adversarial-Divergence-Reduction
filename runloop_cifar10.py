import subprocess 
import numpy as np 
import os 
import sys

os.chdir('./')
ST = 'python '

stand = dict()
conf = dict()

stand = dict()
stand['ds'] = 'cifar10' 
stand['model'] = 'nonwide' 
stand['defense'] = 'none'

conf['adv'] = stand.copy() 
conf['adv']['model'] = 'nonwide'
conf['adv']['defense'] = 'adv'
conf['adv']['perturb'] = 'pgd'

conf['adr_adv'] = stand.copy() 
conf['adr_adv']['model'] = 'nonwide'
conf['adr_adv']['defense'] = 'adr_adv'
conf['adr_adv']['perturb'] = 'pgd'
conf['adr_adv']['lccomw'] = 1.0
conf['adr_adv']['confw'] = 1.0
conf['adr_adv']['vatw'] = 1.0
conf['adr_adv']['gbcomw'] = 1.0
conf['adr_adv']['gbsmtw'] = 1.0


scres = ['train_compact.py', 'eval_compact.py', 'eval_multi_targeted.py']
for k in list(conf.keys()):

	for scr in scres:

		exp = conf[k]
		sub = ' '.join(['--{}={}'.format(t, exp[t]) for t in exp.keys()])
		print(sub)
		subprocess.call([ST + scr + ' ' + sub], shell=True)
