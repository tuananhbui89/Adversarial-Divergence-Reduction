import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from models import switch_status
from compact_loss_pt import local_loss, global_loss
from compact_loss_pt import mysoftmax_cross_entropy_with_two_logits as softmax_xent_two

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def adr_pgd(model,
                x_natural,
                y,
                device,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                projecting=True,
                distance='l_inf', 
                x_min=0.0, 
                x_max=1.0, 
                lccomw=1.0, lcsmtw=1.0,
                gbcomw=1.0, gbsmtw=1.0, 
                confw=1.0):
    assert(beta == 1.0)
    assert(distance == 'l_inf')
    assert(projecting is True)
    assert(x_max > x_min)

    model.eval()

    # random initialization 
    x_adv = Variable(x_natural.data, requires_grad=True)
    random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-epsilon, epsilon).to(device)
    x_adv = Variable(x_adv.data + random_noise, requires_grad=True)

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = nn.CrossEntropyLoss(size_average=False)(model(x_adv), y) # Will not take average over batch 

        grad = torch.autograd.grad(loss_ce, [x_adv])[0] # []
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        if projecting:
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, x_min, x_max)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    nat_output, nat_z = model(x_natural, return_z=True)
    adv_output, adv_z = model(x_adv, return_z=True)

    # Cross entropy loss 
    loss_natural = F.cross_entropy(nat_output, y, reduction='mean') # [b,]
    loss_robust = F.cross_entropy(adv_output, y, reduction='mean') # [b, ]

    # Local compactness 
    lc_com, lc_smt = local_loss(y, nat_z, nat_output, y, adv_z, adv_output)
    
    # Global compactness 
    gb_com, gb_smt = global_loss(y, nat_z, nat_output, y, adv_z, adv_output, scale=0.99)

    # Confidence 
    loss_conf = softmax_xent_two(adv_output, adv_output, reduction='mean')
        
    loss = loss_natural + beta * loss_robust
    loss += lccomw*lc_com + lcsmtw*lc_smt
    loss += gbcomw*gb_com + gbsmtw*gb_smt
    loss += confw*loss_conf

    return loss, x_adv

def adr_trades(model,
                x_natural,
                y,
                device,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                projecting=True,
                distance='l_inf', 
                x_min=0.0, 
                x_max=1.0, 
                lccomw=1.0, lcsmtw=1.0,
                gbcomw=1.0, gbsmtw=1.0, 
                confw=1.0):
    # define KL-loss
    assert(projecting is True)
    assert(distance == 'l_inf')
    assert(x_max > x_min)

    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        if projecting:
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, x_min, x_max)
    
    model.train()

    x_adv = Variable(torch.clamp(x_adv, x_min, x_max), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    nat_output, nat_z = model(x_natural, return_z=True)
    adv_output, adv_z = model(x_adv, return_z=True)
    loss_natural = F.cross_entropy(nat_output, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_output, dim=1),
                                                    F.softmax(nat_output, dim=1))

    # Local compactness 
    lc_com, lc_smt = local_loss(y, nat_z, nat_output, y, adv_z, adv_output)
    
    # Global compactness 
    gb_com, gb_smt = global_loss(y, nat_z, nat_output, y, adv_z, adv_output, scale=0.99)

    # Confidence 
    loss_conf = softmax_xent_two(adv_output, adv_output, reduction='mean')
        
    loss = loss_natural + beta * loss_robust
    loss += lccomw*lc_com + lcsmtw*lc_smt
    loss += gbcomw*gb_com + gbsmtw*gb_smt
    loss += confw*loss_conf

    return loss, x_adv