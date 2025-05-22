#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(Optimizer, lr_decay_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=lr_decay_interval, gamma=lr_decay)

	lr_step = 'epoch'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step