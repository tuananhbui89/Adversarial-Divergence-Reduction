import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.optim as optim
from functools import partial

from trades import trades_loss
from pgd import pgd_loss, pgd_attack, pgd_attack_l2
from adr import adr_pgd, adr_trades
from utils import count_pred 

def get_diff(X, X_adv, order, epsilon=None): 
    X = torch.reshape(X, [X.shape[0], -1])
    X_adv = torch.reshape(X_adv, [X_adv.shape[0], -1])
    d = torch.norm(X_adv - X, p=order, dim=-1, keepdim=True) # [b,]
    if epsilon is not None: 
        delta = torch.abs(X_adv - X) 
        num_exceed = torch.sum(delta > epsilon, dim=1)/X.shape[1]
        num_exceed = torch.mean(num_exceed)
        d = torch.mean(d) # []
        return d, num_exceed
    else: 
        d = torch.mean(d) # []
        return d 

def train(model, data_loader, epoch, optimizer, device, log_interval, attack_params, writer): 
    model.train()
    num_batches = len(data_loader.dataset) // 128

    for batch_idx, (data, target) in enumerate(data_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='none')
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        nat_acc = get_acc(output, target)

        if batch_idx % log_interval == 0:    
            writestr = [
                ('Train_iter={}', epoch*num_batches + batch_idx),
                ('loss={:.4f}', loss.item()), 
                ('nat_acc={:.4f}', nat_acc.item()), 

            ]
            writestr = '  ,'.join([t.format(v) for (t, v) in writestr]) 
            print(writestr)
            writer.add_scalar('loss', loss.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('nat_acc', nat_acc.item(), epoch*num_batches + batch_idx)

    return writer

def baseline_train(model, data_loader, epoch, optimizer, device, log_interval, attack_params, writer): 
    model.train()

    if attack_params['defense'] == 'pgd_train': 
        defense = pgd_loss 
    elif attack_params['defense'] == 'trades_train': 
        defense = trades_loss
    else:
        raise ValueError 

    num_batches = len(data_loader.dataset) // 128

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        loss, X_adv = defense(model=model,
                           x_natural=data,
                           y=target,
                           device=device,
                           optimizer=optimizer,
                           step_size=attack_params['step_size'],
                           epsilon=attack_params['epsilon'],
                           perturb_steps=attack_params['num_steps'],
                           beta=attack_params['trades_beta'], 
                           projecting=attack_params['projecting'], 
                           x_min=attack_params['x_min'], 
                           x_max=attack_params['x_max'])


        loss.backward()
        optimizer.step()

        nat_output = model(data)
        adv_output = model(X_adv)
        nat_acc = get_acc(nat_output, target)
        adv_acc = get_acc(adv_output, target)

        if batch_idx % log_interval == 0:

            writestr = [
                ('Train_iter={}', epoch*num_batches + batch_idx),
                ('nat_acc={:.4f}', nat_acc.item()), 
                ('adv_acc={:.4f}', adv_acc.item()), 
                ('loss={:.4f}', loss.item()), 
            ]
            writestr = '  ,'.join([t.format(v) for (t, v) in writestr]) 
            print(writestr)
            writer.add_scalar('nat_acc', nat_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('adv_acc', adv_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('loss', loss.item(), epoch*num_batches + batch_idx)      
    return writer

def adr_train(model, data_loader, epoch, optimizer, device, log_interval, attack_params, writer): 
    model.train()

    if attack_params['defense'] == 'adr_pgd': 
        defense = adr_pgd
    elif attack_params['defense'] == 'adr_trades': 
        defense = adr_trades
    else:
        raise ValueError 

    num_batches = len(data_loader.dataset) // 128

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        loss, X_adv = defense(model=model,
                           x_natural=data,
                           y=target,
                           device=device,
                           optimizer=optimizer,
                           step_size=attack_params['step_size'],
                           epsilon=attack_params['epsilon'],
                           perturb_steps=attack_params['num_steps'],
                           beta=attack_params['trades_beta'], 
                           projecting=attack_params['projecting'], 
                           x_min=attack_params['x_min'], 
                           x_max=attack_params['x_max'], 
                           lccomw=attack_params['lccomw'], 
                           lcsmtw=attack_params['lcsmtw'], 
                           gbcomw=attack_params['gbcomw'], 
                           gbsmtw=attack_params['gbsmtw'], 
                           confw=attack_params['confw'])


        loss.backward()
        optimizer.step()

        nat_output = model(data)
        adv_output = model(X_adv)
        nat_acc = get_acc(nat_output, target)
        adv_acc = get_acc(adv_output, target)

        if batch_idx % log_interval == 0:

            writestr = [
                ('Train_iter={}', epoch*num_batches + batch_idx),
                ('nat_acc={:.4f}', nat_acc.item()), 
                ('adv_acc={:.4f}', adv_acc.item()), 
                ('loss={:.4f}', loss.item()), 
            ]
            writestr = '  ,'.join([t.format(v) for (t, v) in writestr]) 
            print(writestr)
            writer.add_scalar('nat_acc', nat_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('adv_acc', adv_acc.item(), epoch*num_batches + batch_idx)
            writer.add_scalar('loss', loss.item(), epoch*num_batches + batch_idx)      
    return writer

def test(model, data_loader, device, return_count=False, num_classes=10): 
    model.eval()
    test_loss = 0
    correct = 0

    pred_as_count = np.zeros(shape=[num_classes,])
    correct_count = np.zeros(shape=[num_classes,])
    class_count = np.zeros(shape=[num_classes,])

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            p, c = count_pred(labels=target, preds=output, num_classes=num_classes)
            pred_as_count += p 
            correct_count += c 
            class_count += count_pred(labels=target, preds=target, num_classes=num_classes)[0]

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    print('pred_as_count: ', pred_as_count)
    print('correct_count: ', correct_count)
    
    if return_count: 
        return accuracy, pred_as_count, correct_count, class_count
    else: 
        return accuracy

def adv_test(model, data_loader, device, attack_params, return_count=False, num_classes=10): 
    model.eval()
    test_loss = 0
    correct = 0

    pred_as_count = np.zeros(shape=[num_classes,])
    correct_count = np.zeros(shape=[num_classes,])
    class_count = np.zeros(shape=[num_classes,]) 

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            X_adv, _ = pgd_attack(model, data, target, device, attack_params, status='eval')
            X_adv = Variable(X_adv.data, requires_grad=False)

            output = model(X_adv)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            p, c = count_pred(labels=target, preds=output, num_classes=num_classes)
            pred_as_count += p 
            correct_count += c 
            class_count += count_pred(labels=target, preds=target, num_classes=num_classes)[0]

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print('\nRobustness evaluation : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    
    if return_count: 
        return accuracy, pred_as_count, correct_count, class_count
    else: 
        return accuracy

def adv_test_l2(model, data_loader, device, attack_params, return_count=False, num_classes=10): 
    model.eval()
    test_loss = 0
    correct = 0

    pred_as_count = np.zeros(shape=[num_classes,])
    correct_count = np.zeros(shape=[num_classes,])
    class_count = np.zeros(shape=[num_classes,]) 

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            X_adv, _ = pgd_attack_l2(model, data, target, device, attack_params, status='eval')
            X_adv = Variable(X_adv.data, requires_grad=False)

            output = model(X_adv)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            p, c = count_pred(labels=target, preds=output)
            pred_as_count += p 
            correct_count += c 
            class_count += count_pred(labels=target, preds=target)[0]

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print('\nRobustness evaluation : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    
    if return_count: 
        return accuracy, pred_as_count, correct_count, class_count
    else: 
        return accuracy

def get_pred(model, data_loader, device): 
    model.eval()
    result = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            output = torch.nn.Softmax()(output)
            result.append(output.cpu().numpy())

    result = np.concatenate(result, axis=0)
    return result

def get_acc(output, target): 
    pred = output.argmax(dim=1, keepdim=True)
    acc = torch.mean(pred.eq(target.view_as(pred)).type(torch.FloatTensor))
    return acc 
