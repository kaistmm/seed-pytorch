from .ddpm_pytorch  import DDPMDiffusionPrior

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionPipeline(nn.Module):
    def __init__(self, diffusion_network, args=None, **kwargs):
        """
        diffusion_network : nn.Module class, Options: ['rdm_mlp']. Only mlp-based networks (rdm_mlp) are supported now.
        """
        super().__init__()
        self.args = args

        """
        `pipeline` should be one the class for diffusion-related pipeline.
        """
        # Only diffusers pipeline is supported now. dalle2 pipeline is not supported yet.
        pipeline_config = {
            'num_train_timesteps':   self.args.train_timesteps,
            'embed_dim':             diffusion_network.dim,
            'predict_type':          self.args.predict_type,
            'loss_type':             self.args.loss_type,
            'conditional_diffusion': self.args.conditional_diffusion,
            'self_cond':             self.args.self_cond,
            'training_clamp_norm':   self.args.training_clamp_norm,
            'init_x0_scale':         self.args.init_x0_scale,
            'sampling_clamp_norm':   self.args.sampling_clamp_norm,
            'sampling_final_norm':   self.args.sampling_final_norm,
            'normalize_type':        self.args.normalize_type,
            'scale':                 self.args.clamp_scale
        }
        
        self.pipeline = DDPMDiffusionPrior(diffusion_network, **pipeline_config)


    def norm_clamp_embed(self, x, scale=None):
        return self.pipeline.norm_clamp_embed(x, scale=scale)


    def forward(self, x_target, x_start, cond=None):
        # Only diffusers pipeline supported.
        # `SEED` support only self-conditioning DDPM without `t_cond`.
        loss, pred_x = self.pipeline(x_target, x_start, cond=cond)
        return loss, pred_x

    def sample(self, x_start, cond=None, args=None):
        sample_config = {
            'sample_timesteps': args.sample_timesteps,
            'use_ddim'        : args.use_ddim,
            'self_cond'       : args.self_cond,
            'scale'           : args.clamp_scale,
        }

        # `SEED` support only self-conditioning DDPM without `t_cond`.
        return self.pipeline.sample(x_start, cond, **sample_config)