#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from .cosine_annealing_warmup_restart import CosineAnnealingWarmupRestarts


def Scheduler(optimizer, 
			  first_cycle_steps : int,
              cycle_mult   		: int = 1,
              max_lr     	    : float = 0.001,
              min_lr 	   		: float = 0.000001,
              warmup_steps 		: int = 0,
              gamma 	   		: float = 1.,
              last_epoch 		: int = -1, **kwargs):

	sche_fn = CosineAnnealingWarmupRestarts(optimizer, 
											first_cycle_steps=first_cycle_steps,
											cycle_mult=cycle_mult,
											max_lr=max_lr,
											min_lr=min_lr,
											warmup_steps=warmup_steps,
											gamma=gamma,
											last_epoch=last_epoch)

	lr_step = 'iteration'

	print('Initialised Warmup-based cosine scheduler (having cycle restarts)')

	return sche_fn, lr_step