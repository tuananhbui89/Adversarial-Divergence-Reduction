# -*- coding: utf-8 -*-
"""
    Robustness Evaluation 
"""
import numpy as np
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from mysetting import * 
from models import activ, get_model
from utils_cm import mkdir_p, writelog, list_dir
from mytrain import test, adv_test


import re 
#------------------------------------------------------
# Dataset preprocessing 
if args.ds == 'mnist':
    num_classes = 10
    x_max = 1.
    x_min = 0.
    transform=transforms.Compose([
            transforms.ToTensor(),
            ])
    train_data = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    test_data = datasets.MNIST('../data', train=False,
                        transform=transform)
    epsilon = 0.3 
    step_size = 0.01 
    epsilon_l2 = 0.5 
    step_size_l2 = 0.01 
    num_steps= 200
    epsilon_range = [0.1, 0.2, 0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.6, 0.7]
    epsilon_l2_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    input_size = (1, 28, 28)

elif args.ds == 'cifar10': 
    num_classes = 10
    x_max = 1.
    x_min = 0.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = datasets.CIFAR10('../data', train=True, download=True,
                        transform=transform_train)
    test_data = datasets.CIFAR10('../data', train=False,
                        transform=transform_test)  

    epsilon = 0.031
    step_size = 0.007
    epsilon_l2 = 0.5 
    step_size_l2 = 1.25/255
    num_steps= 200
    epsilon_range = [2., 4., 6., 8., 10., 12., 14., 16., 20., 24., 32.]
    epsilon_range = [x/255 for x in epsilon_range]
    epsilon_l2_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    input_size = (3, 32, 32)


#------------------------------------------------------
# Params setting 

eval_params = dict()

eval_params['attack_type'] = args.eval_attack_type
eval_params['epsilon'] = epsilon #args.epsilon
eval_params['step_size'] = step_size #args.step_size
eval_params['num_steps'] = num_steps #args.num_steps
eval_params['x_min'] = x_min
eval_params['x_max'] = x_max
eval_params['random_start'] = True 
eval_params['dis_input'] = args.dis_input 

eval_params['defense'] = args.defense
eval_params['order'] = np.inf
eval_params['loss_type'] = args.eval_loss_type
eval_params['random_init'] = args.eval_random_init
eval_params['projecting'] = args.eval_projecting
eval_params['distype'] = args.distype

eval_params['trades_beta'] = args.trades_beta

eval_params_l2 = eval_params.copy()
eval_params_l2['epsilon'] = epsilon_l2 #args.epsilon
eval_params_l2['step_size'] = step_size_l2 #args.step_size
eval_params_l2['order'] = 2
# ------------------------------------------------------
import os 
os.chdir('./')
WP = os.path.dirname(os.path.realpath('__file__')) +'/log/'

print(WP)

save_dir = WP + basedir + '/' + modeldir + '/'
mkdir_p(basedir)
mkdir_p(save_dir)

if args.eval_best:
    model_dir = save_dir + 'model_best.pt'
else:
    model_dir = save_dir + 'model.pt'

logfile = save_dir + 'log.txt'
writer = SummaryWriter(save_dir+'log/')

if args.eval_epoch > -1: 
    model_dir = save_dir + 'model-nn-epoch{}.pt'.format(args.eval_epoch)
    writelog('** Test with model_dir:{}'.format(model_dir), logfile)

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(20212022)

device = torch.device("cuda" if use_cuda else "cpu")

test_kwargs = {'batch_size': args.bs}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': False}
    test_kwargs.update(cuda_kwargs)

#------------------------------------------------------
# Load dataset 
train_loader = torch.utils.data.DataLoader(train_data, **test_kwargs)  
test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)    

#------------------------------------------------------
# Model 
model = get_model(args.ds, args.model, activation=activ(args.activ))
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)

summary(model, input_size)

print('** loading model_dir:', model_dir)
model.load_state_dict(torch.load(model_dir))
model.eval()


#------------------------------------------------------
writelog('----------EVAL STANDARD PGD-200 ----------------', logfile)
writelog('model_dir:{}'.format(model_dir), logfile)
nat_acc = test(model, test_loader, device, return_count=False, num_classes=num_classes)
adv_acc = adv_test(model, test_loader, device, eval_params, return_count=False, num_classes=num_classes)
writelog('nat_acc:{}, adv_acc:{}'.format(nat_acc, adv_acc), logfile)

for key in eval_params.keys(): 
    writelog('eval_params, {}:{}'.format(key, eval_params[key]), logfile)
writelog('nat_acc={:.4f}, adv_acc={:.4f}'.format(nat_acc, adv_acc), logfile)
writelog('--------------------------', logfile)

#------------------------------------------------------
if args.eval_multi:
    writelog('----------EVAL MULTIPLE EPSILONS ----------------', logfile)
    writelog('model_dir:{}'.format(model_dir), logfile)
    eval_multi = eval_params.copy()
    for eps in epsilon_range:
        eval_multi['epsilon'] = eps
        eval_multi['num_steps'] = 50 
        # nat_acc = test(model, test_loader, device)
        adv_acc = adv_test(model, test_loader, device, eval_multi, num_classes=num_classes)
        writelog('--------------------------', logfile)
        for key in eval_multi.keys(): 
            writelog('eval_params, {}:{}'.format(key, eval_multi[key]), logfile)
        writelog('nat_acc={:.4f}, adv_acc={:.4f}'.format(nat_acc, adv_acc), logfile)

#------------------------------------------------------
if args.eval_auto: 
    from autoattack import AutoAttack
    writelog('----------EVAL AUTO ATTACK, Standard, Norm Linf ----------------', logfile)
    writelog('model_dir:{}'.format(model_dir), logfile)
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    adversary = AutoAttack(model, norm='Linf', eps=epsilon, log_path=logfile, version='standard') 
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)
    aa_data = TensorDataset(x_adv, y_test)
    aa_loader = DataLoader(aa_data, **test_kwargs)
    adv_acc = test(model, aa_loader, device, num_classes=num_classes)
    writelog('Auto-Attack, Standard, Linf, eps={}, adv_acc={:.4f}'.format(epsilon, adv_acc), logfile)

#------------------------------------------------------
if args.eval_multi_auto: 
    from autoattack import AutoAttack
    writelog('----------EVAL MULTI AUTO ATTACK, Standard, Norm Linf ----------------', logfile)
    writelog('model_dir:{}'.format(model_dir), logfile)
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    for eps in epsilon_range:
        writelog('-------- CHANGE EPS ----------', logfile)
        adversary = AutoAttack(model, norm='Linf', eps=eps, log_path=logfile, version='standard') 
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)
        aa_data = TensorDataset(x_adv, y_test)
        aa_loader = DataLoader(aa_data, **test_kwargs)
        adv_acc = test(model, aa_loader, device, num_classes=num_classes)
        writelog('Auto-Attack, Standard, Linf, eps={}, adv_acc={:.4f}'.format(eps, adv_acc), logfile)

    # writelog('----------EVAL AUTO ATTACK, Standard, Norm L2 ----------------', logfile)
    # adversary = AutoAttack(model, norm='L2', eps=epsilon_l2, log_path=logfile, version='standard') 
    # x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)
    # aa_data = TensorDataset(x_adv, y_test)
    # aa_loader = DataLoader(aa_data, **test_kwargs)
    # adv_acc = test(model, aa_loader, device)
    # writelog('Auto-Attack, Standard, L2, eps={}, adv_acc={:.4f}'.format(epsilon_l2, adv_acc), logfile)
#------------------------------------------------------
if args.eval_bb:
    import foolbox as fb
    writelog('----------EVAL B&B Attack ----------------', logfile)
    writelog('model_dir:{}'.format(model_dir), logfile)
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    class init_attack(object):
        
        def __init__(self, attack):
            self.attack = attack
            
        def run(self, model, originals, criterion_):
            return self.attack(model, x_test, criterion=criterion_, epsilons=epsilon)[1]

    pdg_init_attack = fb.attacks.LinfPGD(steps=20, abs_stepsize=epsilon/2, random_start=True)
    bb_attack = fb.attacks.LinfinityBrendelBethgeAttack(init_attack(pdg_init_attack), steps=200)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    _, _, init_success = pdg_init_attack(fmodel, x_test, y_test, epsilons=epsilon)
    _, advs, success = bb_attack(fmodel, x_test, 
                                criterion=fb.criteria.Misclassification(y_test), 
                                epsilons=epsilon)
    init_acc = 1 - np.mean(init_success.cpu().detach().numpy())
    adv_acc = 1 - np.mean(success.cpu().detach().numpy())
    writelog('B&B Attack, init with PGD-20, eps={}, adv_acc of initial PGD attack={:.4f}'.format(epsilon, init_acc), logfile)
    writelog('B&B Attack, eps={}, adv_acc={:.4f}'.format(epsilon, adv_acc), logfile)

#------------------------------------------------------
if args.eval_multi_bb:
    import foolbox as fb
    writelog('----------EVAL Multi - B&B Attack ----------------', logfile)
    writelog('model_dir:{}'.format(model_dir), logfile)
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    class init_attack(object):
        
        def __init__(self, attack):
            self.attack = attack
            
        def run(self, model, originals, criterion_):
            return self.attack(model, x_test, criterion=criterion_, epsilons=epsilon)[1]

    for eps in epsilon_range:
        pdg_init_attack = fb.attacks.LinfPGD(steps=20, abs_stepsize=eps/2, random_start=True)
        bb_attack = fb.attacks.LinfinityBrendelBethgeAttack(init_attack(pdg_init_attack), steps=200)
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        _, _, init_success = pdg_init_attack(fmodel, x_test, y_test, epsilons=eps)
        _, advs, success = bb_attack(fmodel, x_test, 
                                    criterion=fb.criteria.Misclassification(y_test), 
                                    epsilons=eps)
        init_acc = 1 - np.mean(init_success.cpu().detach().numpy())
        adv_acc = 1 - np.mean(success.cpu().detach().numpy())
        writelog('B&B Attack, init with PGD-20, eps={}, adv_acc of initial PGD attack={:.4f}'.format(eps, init_acc), logfile)
        writelog('B&B Attack, eps={}, adv_acc={:.4f}'.format(eps, adv_acc), logfile)

#------------------------------------------------------
if args.eval_scan: 
    writelog('----------EVAL SCAN ALL CHECKPOINTS ----------------', logfile)
    eval_scan = eval_params.copy()
    all_dirs = list_dir(save_dir, '.pt')
    for cur_dir in all_dirs:
        writelog('--------------------------', logfile)
        writelog('cur_dir:{}'.format(cur_dir), logfile)
        model.load_state_dict(torch.load(cur_dir))
        model.eval()    
        nat_acc = test(model, test_loader, device, num_classes=num_classes)
        adv_acc = adv_test(model, test_loader, device, eval_scan, num_classes=num_classes)
        for key in eval_scan.keys(): 
            writelog('eval_params, {}:{}'.format(key, eval_scan[key]), logfile)
        writelog('nat_acc={:.4f}, adv_acc={:.4f}'.format(nat_acc, adv_acc), logfile)


