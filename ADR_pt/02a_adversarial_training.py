"""
    Adversarial Training for MNIST and CIFAR dataset
"""
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

from mysetting import * 
from models import activ, get_model, adjust_learning_rate, get_optimizer
from utils_cm import mkdir_p, writelog, backup
from mytrain import train, test, adv_test
from mytrain import baseline_train

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
    num_steps= 40
    log_period = 10
    epsilon_range = [0.1, 0.2, 0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.6, 0.7]

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
    num_steps= 10
    log_period = 50 if args.model == 'wideresnet' else 20 
    epsilon_range = [2., 4., 6., 8., 10., 12., 14., 16., 20., 24., 32.]
    epsilon_range = [x/255 for x in epsilon_range]

#------------------------------------------------------
# Params setting 
log_interval = 10

attack_params = dict()
attack_params['attack_type'] = args.attack_type
attack_params['epsilon'] = args.epsilon
attack_params['step_size'] = step_size if args.step_size < 0 else args.step_size
attack_params['num_steps'] = num_steps
attack_params['x_min'] = x_min
attack_params['x_max'] = x_max
attack_params['ls_factor'] = args.ls_factor

attack_params['defense'] = args.defense
attack_params['order'] = int(args.order) if args.order != 'inf' else np.inf
attack_params['loss_type'] = args.loss_type
attack_params['random_init'] = args.random_init
attack_params['projecting'] = args.projecting
attack_params['distype'] = args.distype
attack_params['trades_beta'] = args.trades_beta
attack_params['lccomw'] = args.lccomw
attack_params['lcsmtw'] = args.lcsmtw
attack_params['gbcomw'] = args.gbcomw
attack_params['gbsmtw'] = args.gbsmtw
attack_params['confw'] = args.confw

eval_params = attack_params.copy()
eval_params['num_steps'] = 20
eval_params['epsilon'] = epsilon
# ------------------------------------------------------
import os 
os.chdir('./')
WP = os.path.dirname(os.path.realpath('__file__')) +'/log/'
print(WP)

save_dir = WP + basedir + '/' + modeldir + '/'
mkdir_p(basedir)
mkdir_p(save_dir)
mkdir_p(save_dir+'/codes/')
backup('./', save_dir+'/codes/')
model_dir = save_dir + 'model.pt'
model_best_dir = save_dir + 'model_best.pt'

logfile = save_dir + 'log.txt'
writer = SummaryWriter(save_dir+'log/')

for key in attack_params.keys(): 
    writelog('attack_params, {}:{}'.format(key, attack_params[key]), logfile)

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(20212022)

device = torch.device("cuda" if use_cuda else "cpu")

train_kwargs = {'batch_size': args.bs, 'shuffle': True} #'drop_last': True
test_kwargs = {'batch_size': args.bs}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

#------------------------------------------------------
# Load dataset 
train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)    

#------------------------------------------------------
# Model 
model = get_model(args.ds, args.model, activation=activ(args.activ))
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

model.to(device)

opt, lr = get_optimizer(ds=args.ds, model=model, architecture=args.model)
#------------------------------------------------------
# Train model 
pre_acc = -1. 

if args.defense == 'none': 
    for epoch in range(args.epochs): 
        opt = adjust_learning_rate(opt, epoch, lr=lr, ds=args.ds)
        writer = train(model, train_loader, epoch, opt, device, log_interval, attack_params, writer)
        nat_acc, nat_pred_as_count, nat_correct_count, class_count = test(model, test_loader, device, return_count=True, num_classes=num_classes)
        if epoch % log_period == 0 and epoch > 0:
            adv_acc, adv_pred_as_count, adv_correct_count, class_count = adv_test(model, test_loader, device, eval_params, return_count=True, num_classes=num_classes)
            writelog('epoch:{}, nat_acc:{}, adv_acc:{}'.format(epoch, nat_acc, adv_acc), logfile)
            writelog('nat_pred_as_count: {}'.format(nat_pred_as_count), logfile)
            writelog('nat_correct_count: {}'.format(nat_correct_count), logfile)
            writelog('adv_pred_as_count: {}'.format(adv_pred_as_count), logfile)
            writelog('adv_correct_count: {}'.format(adv_correct_count), logfile)
            writelog('class_count: {}'.format(class_count), logfile)

            if (nat_acc + adv_acc) >= pre_acc: 
                pre_acc = nat_acc + adv_acc 
                torch.save(model.state_dict(), model_best_dir)
        else: 
            writelog('epoch:{}, nat_acc:{}'.format(epoch, nat_acc), logfile)
            writelog('nat_pred_as_count: {}'.format(nat_pred_as_count), logfile)
            writelog('nat_correct_count: {}'.format(nat_correct_count), logfile)
            writelog('class_count: {}'.format(class_count), logfile)

        torch.save(model.state_dict(), model_dir)
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'model-nn-epoch{}.pt'.format(epoch)))
        writer.flush()
    writer.close()  

else:
    assert (args.defense in ['pgd_train', 'trades_train'])
    for epoch in range(args.epochs): 
        opt = adjust_learning_rate(opt, epoch, lr=lr, ds=args.ds)
        writer = baseline_train(model, train_loader, epoch, opt, device, log_interval, attack_params, writer)

        nat_acc, nat_pred_as_count, nat_correct_count, class_count = test(model, test_loader, device, return_count=True, num_classes=num_classes)
        if epoch % log_period == 0 and epoch > 0:
            adv_acc, adv_pred_as_count, adv_correct_count, class_count = adv_test(model, test_loader, device, eval_params, return_count=True, num_classes=num_classes)
            writelog('epoch:{}, nat_acc:{}, adv_acc:{}'.format(epoch, nat_acc, adv_acc), logfile)
            writelog('nat_pred_as_count: {}'.format(nat_pred_as_count), logfile)
            writelog('nat_correct_count: {}'.format(nat_correct_count), logfile)
            writelog('adv_pred_as_count: {}'.format(adv_pred_as_count), logfile)
            writelog('adv_correct_count: {}'.format(adv_correct_count), logfile)
            writelog('class_count: {}'.format(class_count), logfile)

            if (nat_acc + adv_acc) >= pre_acc: 
                pre_acc = nat_acc + adv_acc 
                torch.save(model.state_dict(), model_best_dir)
        else: 
            writelog('epoch:{}, nat_acc:{}'.format(epoch, nat_acc), logfile)
            writelog('nat_pred_as_count: {}'.format(nat_pred_as_count), logfile)
            writelog('nat_correct_count: {}'.format(nat_correct_count), logfile)
            writelog('class_count: {}'.format(class_count), logfile)

        torch.save(model.state_dict(), model_dir)
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'model-nn-epoch{}.pt'.format(epoch)))
        writer.flush()
    writer.close()   


nat_acc, nat_pred_as_count, nat_correct_count, class_count  = test(model, test_loader, device, return_count=True, num_classes=num_classes)
adv_acc, adv_pred_as_count, adv_correct_count, class_count = adv_test(model, test_loader, device, eval_params, return_count=True, num_classes=num_classes)
writelog('epoch:{}, nat_acc:{}, adv_acc:{}'.format(epoch, nat_acc, adv_acc), logfile)
writelog('nat_pred_as_count: {}'.format(nat_pred_as_count), logfile)
writelog('nat_correct_count: {}'.format(nat_correct_count), logfile)
writelog('adv_pred_as_count: {}'.format(adv_pred_as_count), logfile)
writelog('adv_correct_count: {}'.format(adv_correct_count), logfile)
writelog('class_count: {}'.format(class_count), logfile)

#------------------------------------------------------
model_dir = model_best_dir
model.load_state_dict(torch.load(model_dir))
model.eval()
eval_params = attack_params.copy()
eval_params['num_steps'] = 200
eval_params['epsilon'] = epsilon

#------------------------------------------------------
writelog('----------EVAL STANDARD PGD-200 ----------------', logfile)
writelog('model_dir:{}'.format(model_dir), logfile)
nat_acc, nat_pred_as_count, nat_correct_count, class_count = test(model, test_loader, device, return_count=True, num_classes=num_classes)
adv_acc, adv_pred_as_count, adv_correct_count, class_count = adv_test(model, test_loader, device, eval_params, return_count=True, num_classes=num_classes)
writelog('epoch:{}, nat_acc:{}, adv_acc:{}'.format(epoch, nat_acc, adv_acc), logfile)
writelog('nat_pred_as_count: {}'.format(nat_pred_as_count), logfile)
writelog('nat_correct_count: {}'.format(nat_correct_count), logfile)
writelog('adv_pred_as_count: {}'.format(adv_pred_as_count), logfile)
writelog('adv_correct_count: {}'.format(adv_correct_count), logfile)
writelog('class_count: {}'.format(class_count), logfile)
writelog('--------------------------', logfile)
for key in eval_params.keys(): 
    writelog('eval_params, {}:{}'.format(key, eval_params[key]), logfile)
writelog('nat_acc={:.4f}, adv_acc={:.4f}'.format(nat_acc, adv_acc), logfile)

