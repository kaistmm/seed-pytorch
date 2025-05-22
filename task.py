#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob, os
import numpy, sys, random, time, itertools, importlib
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler

from models.diffusion.networks.mlp.rdm_mlp        import RDMMLP
from models.diffusion.pipeline.diffusion_pipeline import DiffusionPipeline

from torch_ema import ExponentialMovingAverage



def normalize(x):
    x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True))
    return x

class WrappedModel(nn.Module):
    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, eval=False):
        return self.module(x, eval)


class TaskModel(nn.Module):
    def __init__(self, backbone, args, **kwargs):
        super(TaskModel, self).__init__()

        # Existing SpeakerNet model and loss function. You can change the model and loss function by your own.
        self.backbone      = importlib.import_module("models.speakernet." + backbone).__getattribute__("MainModel")(**kwargs)

        # Freezing the backone network
        for param in self.backbone.parameters(): param.requires_grad = False
        self.backbone.eval()

        # Diffusion pipeline
        self.embedding_size = self.backbone.nOut
        self.train_diffusion = args.train_diffusion

        self.diffusion_network = RDMMLP(
            in_channels=self.embedding_size,
            time_embed_dim=self.embedding_size,
            model_channels=self.embedding_size*2,
            bottleneck_channels=self.embedding_size*2,
            out_channels=self.embedding_size,
            num_res_blocks=args.diffusion_num_layers,
            dropout=0,
            use_context=args.conditional_diffusion,
            context_channels=self.embedding_size
        )

        # The pipeline of Diffusion
        self.diffusion_pipeline = DiffusionPipeline(
            diffusion_network  = self.diffusion_network,
            diffusion_pipeline = args.diffusion_pipeline,
            args = args
        )

    def forward(self, data, eval=False):
        if len(data.size()) == 3:
            S, B, D = data.size()

        with torch.no_grad():
            self.backbone.eval()
            data = data.reshape(-1, data.size()[-1]).cuda()
            spk_embeds = self.backbone.forward(data).detach()

        result = {
            'speaker_embedding': spk_embeds
        }

        if eval: # If you need only the speaker embedding for evaluation
            return result
        else:
            assert self.train_diffusion, "You need to set train_diffusion to True to train the diffusion model"


            spk_embeds = spk_embeds.reshape(S, B, spk_embeds.size()[-1])             # Reshape, [S*B,D] -> [S,B,D]
            spk_target_embed, spk_paired_embed = spk_embeds[0], spk_embeds[1:]       # Split the target and input embed, [B,D], [S-1,B,D]
            # spk_paired_embed -> speaker (clean + noisy) embeddings

            # repeat original embedding to match the shape of paired (clean + noisy) embeddings
            x_target = spk_target_embed.unsqueeze(0).expand(spk_paired_embed.size(0), -1, -1)    # Expand shape, [B,D] -> [S-1,B,D]
            x_start  = spk_paired_embed

            x_target = x_target.reshape(-1, x_target.size()[-1])   # Reshape, [(S-1)*B,D]
            x_start  = x_start.reshape(-1, x_start.size()[-1])       # Reshape, [(S-1)*B,D]

            nloss_diff, pred_embed = self.diffusion_pipeline(x_target=x_target, x_start=x_start)

            #pred_embed = pred_embed.reshape(S-1, B, pred_embed.size()[-1]) # Reshape, [((S-1)*B),D] -> [S-1,B,D]

            return {'diffusion_loss': nloss_diff}


class ModelTrainer(object):
    def __init__(self, task_model, optimizer, scheduler, gpu, mixedprec, grad_clip, grad_clip_type, ema, **kwargs):

        self.task = task_model

        self.backbone           = self.task.module.backbone
        self.diffusion_network  = self.task.module.diffusion_network
        self.diffusion_pipeline = self.task.module.diffusion_pipeline

        self.ema = ExponentialMovingAverage(self.task.parameters(), decay=ema) if ema > 0 else None


        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.task.parameters(), **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler()

        self.gpu = gpu

        self.mixedprec = mixedprec

        self.grad_clip = grad_clip
        self.grad_clip_type = grad_clip_type

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose, args, lr_schedule=False, logger=None):
        self.task.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss, lossDM = 0, 0 
        top1 = 0
        # EER or accuracy
        iters = len(loader)

        tstart = time.time()

        for idx, data in enumerate(loader):
            
            data = data.transpose(1, 0) # [B, S, D] -> [S, B, D]

            self.task.zero_grad()

            with autocast(enabled=self.mixedprec):
                outputs = self.task(data)
                
                nloss_diff = outputs['diffusion_loss']
                nloss      = nloss_diff

            if self.mixedprec:
                self.scaler.scale(nloss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.task.parameters(), self.grad_clip, self.grad_clip_type)
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.task.parameters(), self.grad_clip, self.grad_clip_type)
                self.__optimizer__.step()

            if self.ema is not None:
                self.ema.update(self.diffusion_network.parameters())

            loss    += nloss.detach().cpu().item()
            lossDM  += nloss_diff.detach().cpu().item()

            counter += 1
            index += stepsize

            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                args.global_step += 1
                lr = self.__optimizer__.param_groups[0]["lr"]

                logger.log({
                    "DM loss_step": loss / counter,
                    "grad_norm": grad_norm,
                    "lr": lr,
                })

                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write("LossDM {:.2f} LR {:.6f} - {:.2f} Hz ".format(lossDM / counter, lr, stepsize / telapsed))
                sys.stdout.flush()

            if self.lr_step == "iteration" and lr_schedule:
                self.__scheduler__.step()


        if self.lr_step == "epoch" and lr_schedule:
            self.__scheduler__.step()

        return (loss / counter, top1 / counter)


    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=5, num_eval=10, args=None, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.task.eval()

        original_diffusion_params_stored_for_eval = False
        if self.ema is not None and self.task.train_diffusion:
            # Store original parameters and apply EMA parameters for evaluation
            self.ema.store(self.diffusion_network.parameters())
            self.ema.copy_to(self.diffusion_network.parameters())
            original_diffusion_params_stored_for_eval = True
            if rank == 0:
                print("\n[Exponential Moving Average activated] Using EMA parameters for diffusion_network during evaluation.")
        
        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        print_interval = max(1, print_interval // args.batch_size)
        batch_size     = max(1, args.batch_size // num_eval)
        test_loader    = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp, labels = data # inp : [B, num_eval, D], labels : [B]
            with torch.no_grad():
                # self.task.forward will use self.diffusion_network,
                # which now has EMA parameters if self.ema was active and setup was successful.
                outputs = self.task(inp, eval=True) 
                spk_embeds = outputs['speaker_embedding'] # [B*num_eval, D]

                # Apply Diffusion if used
                if args.train_diffusion:
                    # The diffusion_pipeline uses self.diffusion_network internally
                    enhanced_spk_embeds = self.diffusion_pipeline.sample(x_start=spk_embeds, args=args)

                    spk_embeds = spk_embeds + enhanced_spk_embeds if args.feature_ensemble else enhanced_spk_embeds

                embed_dims = spk_embeds.size()[-1]
                spk_embeds = spk_embeds.reshape(-1, num_eval, embed_dims).detach().cpu() # [B, num_eval, D]
                
            for i, label in enumerate(labels): # Corrected variable name from idx to i to avoid conflict
                feats[label] = spk_embeds[i]
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), (idx+1) * test_loader.batch_size / telapsed, spk_embeds.size()[2]) # Corrected embedding size access and Hz calculation
                )
                sys.stdout.flush()
        
        if original_diffusion_params_stored_for_eval:
            # Restore original parameters to the diffusion_network
            self.ema.restore(self.diffusion_network.parameters())
            if rank == 0:
                print("\n[Exponential Moving Average activated] Restored original parameters to diffusion_network after evaluation.")

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        eer, mindcf, threshold = None, None, None
        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):
                data = line.split() 

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]].cuda().float()
                com_feat = feats[data[2]].cuda().float()

                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = torch.cdist(ref_feat.reshape(-1, embed_dims), com_feat.reshape(-1, embed_dims)).detach().cpu().numpy()
                score = -1 * numpy.mean(dist)

                #dist = torch.nn.functional.cosine_similarity(ref_feat.reshape(-1, embed_dims), com_feat.reshape(-1, embed_dims)).detach().cpu().numpy()
                #score = numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

            result = tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])
            eer    = result[1]
            fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

        return (eer, mindcf, threshold)
        
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def save_parameters(self, target_model:str, path:str):
        if target_model == "backbone":
            model = self.backbone
        elif target_model == "diffusion_network":
            model = self.diffusion_network
        else:
            raise ValueError(f"Invalid target model: {target_model}")

        torch.save(model.state_dict(), path)

        if self.ema is not None and target_model == "diffusion_network":
            ema_path = path.replace(".pth", "_ema.pth")
            self.ema.store(self.diffusion_network.parameters())
            self.ema.copy_to(self.diffusion_network.parameters())
            torch.save(self.diffusion_network.state_dict(), ema_path)
            self.ema.restore(self.diffusion_network.parameters())


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def load_parameters(self, target_model:str, path:str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        state_dict = torch.load(path, map_location="cuda:%d" % self.gpu)

        if target_model == "backbone":
            model = self.backbone
        elif target_model == "diffusion_network":
            model = self.diffusion_network
        else:
            raise ValueError(f"Invalid target model: {target_model}")

        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        print(f"{model.__class__.__name__} Model loaded from {path}")
        
        # Report any parameter mismatches with full details
        if incompatible_keys.missing_keys:
            print(f"Missing keys ({len(incompatible_keys.missing_keys)}) of {model.__class__.__name__}:")
            for key in incompatible_keys.missing_keys:
                print(f"  - {key}")
        
        if incompatible_keys.unexpected_keys:
            print(f"Unexpected keys ({len(incompatible_keys.unexpected_keys)}) from {os.path.basename(path)}:")
            for key in incompatible_keys.unexpected_keys:
                print(f"  - {key}")
            

"""
Metric code for Speaver Verification Task
"""
def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    
    return (tunedThreshold, eer, fpr, fnr);

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold