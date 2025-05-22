#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import glob
import zipfile
import warnings
import datetime
import wandb

import torch

import torch.distributed as dist
import torch.multiprocessing as mp

from task import *
from DatasetLoader import *

warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "Pytorch Framework for `SEED: Speaker Embedding Enhancement Diffusion` paper.")

parser.add_argument('--config',             type=str,   default=None,       help='Config YAML file')
parser.add_argument('--max_frames',         type=int,   default=200,        help='Input length to the network for training')
parser.add_argument('--eval_frames',        type=int,   default=400,        help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--num_eval',           type=int,   default=10,         help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',         type=int,   default=200,        help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk',    type=int,   default=1500,       help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread',  type=int,   default=8,          help='Number of loader threads')
parser.add_argument('--augment',            type=bool,  default=True,       help='Augment input')
parser.add_argument('--augment-8k',         type=bool,  default=False,      help='Augment 8k input')
parser.add_argument('--specaug',            action='store_true',            help='Augment input')
parser.add_argument('--seed',               type=int,   default=random.randint(0, 10**4),  help='The seed value ~ [0, 10000]. It effects for the diffusion sampling process. If you want to get same sampling results, set the same seed value.')

## Wandb logger
parser.add_argument('--project',            default='SEED',                 help='Main project name and Wandb project name.')
parser.add_argument('--entity',             default='seed_guys',            help='Wandb entity name.')
parser.add_argument('--group',              default='seed_tuning',          help='Wandb group name.')
parser.add_argument('--name',               default='baseline',             help='Experiment name.')
parser.add_argument('--description',        default='')

## Training details
parser.add_argument('--test_interval',      type=int,   default=1,      help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',          type=int,   default=300,    help='Maximum number of epochs')

## Optimizer
parser.add_argument('--optimizer',          type=str,   default="adam",     help='sgd or adam')
parser.add_argument('--scheduler',          type=str,   default="steplr",   help='Learning rate scheduler')
parser.add_argument('--lr',                 type=float, default=0.001,      help='Learning rate')
parser.add_argument('--weight_decay',       type=float, default=5e-5,       help='Weight decay in the optimizer')
parser.add_argument('--grad_clip',          type=float, default=9999,       help='Gradient clipping value')
parser.add_argument('--grad_clip_type',     type=float, default=2.0,        help='Gradient clipping type')

# For linear lr annealing 
parser.add_argument("--lr_decay",           type=float, default=0.97,       help='Learning rate decay every [lr_decay_interval] epochs')
parser.add_argument("--lr_decay_interval",  type=int,   default=1,          help='Learning rate decay interval')
parser.add_argument("--lr_decay_start",     type=int,   default=0,          help='Learning rate decay start epoch')


## Diffusion
parser.add_argument('--train_diffusion',        type=bool,  default=False,       help='Use diffusion model')
parser.add_argument('--diffusion_network',      type=str,   default='rdm_mlp',   help='Type of diffusion network. Options: [rdm_mlp]. Only for 1D diffusion yet')
parser.add_argument('--diffusion_num_layers',   type=int,   default=3,           help='Number of layers for diffusion network')
parser.add_argument('--diffusion_pipeline',     type=str,   default='diffusers', help='Type of diffusion prior pipeline. Options: [dalle2, diffusers]')
parser.add_argument('--train_timesteps',        type=int,   default=1000,        help='Total training timesteps for noise scheduler')
parser.add_argument('--sample_timesteps',       type=int,   default=50,          help='Total inference timesteps for noise scheduler')
parser.add_argument('--conditional_diffusion',  type=bool,  default=False,       help='Use conditional diffusion')
parser.add_argument('--loss_type',              type=str,   default='l1',        help='loss function type of diffusion model. Options: [l2, l1, smooth_l1]')
parser.add_argument('--predict_type',           type=str,   default='sample',    help='Type of prediction. [epsilon, v_prediction, sample]')
parser.add_argument('--self_cond',              type=bool,  default=False,       help='Use self condition for diffusion model. If True, use x_cond as x_start')
parser.add_argument('--training_clamp_norm',    type=bool,  default=False,       help='If True, use `norm_clamp_embed()` for all embeddings in training')
parser.add_argument('--init_x0_scale',          type=bool,  default=False,       help='If True, scaling with `clamp_scale` value for all embeddings in initialx_start')
parser.add_argument('--sampling_clamp_norm',    type=bool,  default=False,       help='If True, use `norm_clamp_embed()` for all embeddings in sampling')
parser.add_argument('--sampling_final_norm',    type=bool,  default=False,       help='If True, use `norm_clamp_embed()` for final output embedding in sampling')
parser.add_argument('--normalize_type',         type=str,   default='l2',        help='Type of normalization.  Options: [l2, meanstd]')
parser.add_argument('--clamp_scale',            type=float, default=None,        help='Scale value for `norm_clamp_embed()`')
parser.add_argument('--use_ddim',               type=bool,  default=False,       help='if True, use DDIM sampling, else predict x_start directly from t-step')
parser.add_argument('--feature_ensemble',       type=bool,  default=False,       help='if True, use feature ensemble with diffusion output and original feature')
parser.add_argument('--ema',                    type=float, default=0,           help='EMA decay rate. e.g. 0.9999. value 0 will disable EMA.')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--K',              type=int,   default=3,      help='K value, only for aamsoftmax_subcenter_intertopk_loss')
parser.add_argument('--mp',             type=float, default=0.06,   help='mp value, only for aamsoftmax_subcenter_intertopk_loss')
parser.add_argument('--top-k',          type=float, default=5,      help='top k value, only for aamsoftmax_subcenter_intertopk_loss')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--pretrained_backbone_model',  type=str,   default="",  help='Pretrained speaker model weights')
parser.add_argument('--pretrained_diffusion_model', type=str,   default="",  help='Pretrained diffusion model weights')
parser.add_argument('--save_path',                  type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="datasets/manifests/train_list.txt",    help='Train list.      Default path style is follow `voxceleb` style like `speaker_id/video_session/filename.wav`')
parser.add_argument('--test_list',      type=str,   default="datasets/manifests/vox1-o.txt",        help='Evaluation list. Default path style is follow `voxceleb` style like `speaker_id/video_session/filename.wav`')
parser.add_argument('--extra_test_list',type=str,   default="datasets/manifests/vcmix_test.txt",    help='Additional Evaluation list')
parser.add_argument('--train_path',     type=str,   default=None,        help='Absolute path to the train set. Default is None. If is not None, dataloadet get data path with train_path + train_list path. e.g `train_path/ + file_path of `train_list`')
parser.add_argument('--test_path',      type=str,   default=None,        help='Absolute path to the test set.  Default is None. If is not None, dataloadet get data path with test_path + test_list path.   e.g `test_path/ + file_path of `test_list`')
parser.add_argument('--musan_path',     type=str,   default="datasets/musan",            help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="datasets/simulated_rirs",   help='Absolute path to the test set')

## Backbone Model parameters (For ECAPA-TDNN and Resnet34SEV2)
parser.add_argument('--backbone',       type=str,   default="ECAPA_TDNN",     help='Name of model definition')
parser.add_argument('--n_mels',         type=int,   default=80,               help='Number of mel or mfcc filterbanks')
parser.add_argument('--log_input',      type=bool,  default=True,             help='Log input features')
parser.add_argument('--encoder_type',   type=str,   default="ECA",            help='Type of encoder')
parser.add_argument('--C',              type=int,   default=1024,             help='Channel size for the speaker encoder')
parser.add_argument('--nOut',           type=int,   default=256,              help='Speaker Embedding size (output embedding of backbone model)')

## For test only
parser.add_argument('--eval',            dest='eval',    action='store_true',    help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args()

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):
    # Set seed unconditionally for this worker process
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 

    args.gpu = gpu

    ## Load models
    task = TaskModel(**vars(args), args=args)

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        task.cuda(args.gpu)

        task = torch.nn.parallel.DistributedDataParallel(task, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        task = WrappedModel(task).cuda(args.gpu)

    # Initialize wandb
    if args.gpu == 0:
        wandb.init(
            project=args.project,
            entity=args.entity,
            group=args.group,
            name=args.name,
            config=vars(args),
        )
        # Log model architecture
        wandb.watch(task)

    it = 1
    eers = [100]

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        shuffle=True,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    
    if args.gpu == 0:
        ## Write args to scorefile
        scorefile = open(args.result_save_path+"/scores.txt", "a+")

        print("Diffusion use:", args.train_diffusion)
        if args.train_diffusion:
            print("Diffusion training  timesteps:",  args.train_timesteps)
            print("Diffusion inference timesteps:",  args.sample_timesteps)
            print("Diffusion with condition:",       args.conditional_diffusion)
            print("Diffusion self condition:",       args.self_cond)
            print("Diffusion training clamp norm:",  args.training_clamp_norm)
            print("Diffusion init x0  scale:     ",  args.init_x0_scale)
            print("Diffusion sampling clamp norm:",  args.sampling_clamp_norm)
            print("Diffusion sampling final norm:",  args.sampling_final_norm)
            print("Diffusion normalize type:     ",  args.normalize_type)
            print("Diffusion sampling by DDIM:   ",  args.use_ddim)
            print("Diffusion feature ensemble:   ",  args.feature_ensemble)
            print("Diffusion network ema: ",         args.ema)

        print("Activate Audio 8k augmentation:", args.augment_8k)
        print("Max Learning rate :",             args.lr)
        print("Seed value:",                     args.seed)

        pytorch_total_params = sum(p.numel() for p in task.module.parameters())
        trainable_params     = sum(p.numel() for p in task.module.parameters() if p.requires_grad)
        frozen_params        = pytorch_total_params - trainable_params

        print('Total parameters:     ',f'{pytorch_total_params/1000000:.2f}M')
        print('Trainable parameters: ',f'{trainable_params/1000000:.2f}M (Diffusion network)')
        print('Frozen parameters:    ',f'{frozen_params/1000000:.2f}M')


    trainer  = ModelTrainer(task, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    # 1. Pretrained speaker model
    it = 0
    if args.pretrained_backbone_model != "":
        trainer.load_parameters("backbone", args.pretrained_backbone_model)

    if args.pretrained_diffusion_model != "":
        trainer.load_parameters("diffusion_network", args.pretrained_diffusion_model)

    """ We didn't implement the continue training code. (TODO)"""


    if trainer.lr_step == 'epoch' and args.scheduler == 'steplr':
        for ii in range(1,it):
            trainer.__scheduler__.step()

    ## Evaluation code - must run on single GPU
    if args.eval == True :
        if args.gpu == 0: print('Test list: ',args.test_list)
        
        eer, mindcf, threshold = trainer.evaluateFromList(**vars(args), args=args)

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(eer), "MinDCF {:2.5f}".format(mindcf))

            test_name = os.path.splitext(os.path.basename(args.test_list))[0]

            wandb.log({
                f"eval/EER_{test_name}": eer,
                f"eval/MinDCF_{test_name}": mindcf,
                "epoch": it
            })

        return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)


    ## Core training script
    test_lists = [args.test_list, args.extra_test_list] if args.extra_test_list else [args.test_list] # you can add more test lists (we recommend two eval sets for better checking)
    args.global_step = 0
    for it in range(it,args.max_epoch+1):
        lr_schedule = True if it+1 > args.lr_decay_start else False
            
        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == 0), args=args, lr_schedule=lr_schedule, logger=wandb if args.gpu == 0 else None)
        # Get learning rate directly from the first param group instead of iterating through all groups
        clr = trainer.__optimizer__.param_groups[0]['lr']

        if args.gpu == 0:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, clr))
            scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, clr))
            
            # Log training metrics to wandb
            wandb.log({
                "train/loss": loss,
                "train/accuracy": traineer,
                "train/learning_rate": clr,
                "epoch": it
            })

        if it % args.test_interval == 0:
            metrics = {}
            wandb_metrics = {}  # Dictionary to collect all metrics for wandb
            
            for test_list in test_lists: 
                test_name = os.path.splitext(os.path.basename(test_list))[0]
                args.test_list = test_list
                eer, mindcf, threshold = trainer.evaluateFromList(**vars(args), args=args)

                if args.gpu == 0:
                    metrics[f'{test_name}/EER'] = eer
                    metrics[f'{test_name}/EER'] = eer
                    metrics[f'{test_name}/mindcf'] = mindcf
                    
                    # Collect metrics for wandb
                    wandb_metrics[f"eval/EER_{test_name}"] = eer
                    wandb_metrics[f"eval/MinDCF_{test_name}"] = mindcf

            if args.gpu == 0:
                # Log all test metrics to wandb at once
                wandb_metrics["epoch"] = it
                wandb.log(wandb_metrics)

                if args.train_diffusion:
                    trainer.save_parameters("diffusion_network", args.model_save_path+"/model%09d.model"%it)
                else:
                    trainer.save_parameters("backbone", args.model_save_path+"/model%09d.model"%it)

            print('\n')
            for test_list in test_lists:
                test_name = os.path.splitext(os.path.basename(test_list))[0]

                eer     = metrics[f'{test_name}/EER']
                mindcf  = metrics[f'{test_name}/mindcf']
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f} TEST: {}".format(it, eer, mindcf, test_name))
                scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f} TEST: {}\n".format(it, eer, mindcf, test_name))

                scorefile.flush()

    if args.gpu == 0:
        scorefile.close()
        # Finish wandb run
        wandb.finish()

## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version: ', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs: ', torch.cuda.device_count())
    print('Save path:      ', args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()