import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm

from diffusers import DDIMScheduler


def normalize(t, normalize_type='l2'):
    # t : (B,...,D) tensor
    if normalize_type == 'l2':
        return F.normalize(t, dim=-1)

    elif normalize_type == 'meanstd':
        mean = t.mean(dim=-1, keepdim=True)
        std = t.std(dim=-1, keepdim=True)
        return (t - mean) / std

    else:
        raise ValueError(f"Unknown normalize type {normalize_type}")


class DDPMDiffusionPrior(nn.Module):
    def __init__(self, 
                 model, 
                 num_train_timesteps,
                 embed_dim,
                 predict_type="epsilon",
                 loss_type="l1",
                 conditional_diffusion=False, # if not use condition for diffusion, prediction x_start with only noisy_x_start and without condition. 
                 self_cond=False,             # If use, input noisy_x_start is based on the condition x. 
                 training_clamp_norm=False,
                 init_x0_scale=False,
                 sampling_clamp_norm=False,
                 sampling_final_norm=False,
                 normalize_type='l2', # ['l2', 'meanstd']
                 scale = None, # if dont want to scale, set 1.0
                 **kargs):
        """
        We follow the `Diffusers` code to create a custom diffusion prior model.
        """
        super().__init__()
        self.model = model
        self.num_train_timesteps = num_train_timesteps
        self.predict_type = predict_type 
        self.loss_type = loss_type
        self.conditional_diffusion = conditional_diffusion
        self.self_cond = self_cond

        # For normalize and scaling the feature or embed of the training and sampling process
        self.training_clamp_norm = training_clamp_norm
        self.init_x0_scale       = init_x0_scale
        self.sampling_clamp_norm = sampling_clamp_norm
        self.sampling_final_norm = sampling_final_norm
        self.normalize_type = normalize_type

        # default scaling is sqrt(D) | @crowsonkb 's suggestion - https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        self.scale = scale if scale is not None else embed_dim ** 0.5  

        losses = {'l1': F.l1_loss, 'l2': F.mse_loss, 'huber': F.smooth_l1_loss}

        if loss_type in losses:
            self.loss_fn = losses[loss_type]
        else:
            assert False, f"Unknown loss type {loss_type} of DDPMDiffusionPrior"

        # We use DDIMScheduler for DDPM model.
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule='scaled_linear',
            prediction_type=predict_type,
            # Generally, the range of the data is [-3, 3] for the guassian distribution. So, we will clip the normalized and scaled output by the range of [-3 * scale, 3 * scale].
            clip_sample_range=self.scale * 3 
        )


    def norm_clamp_embed(self, t, scale=None):
        if scale is None: scale = self.scale

        t = normalize(t, self.normalize_type) * scale
        return t


    def step(self, model_output, timesteps, sample):
        """
        model_output : torch.Tensor, (B, ...)
        timesteps : torch.Tensor, (B,)
        sample : torch.Tensor, (B, ...)
        """
        # self.noise_scheduler.step(pred, timesteps, x_t) 을 참고함.
        alpha_t = self.noise_scheduler.alphas_cumprod[timesteps] # (B,)
        beta_t = 1 - alpha_t # (B,)

        while len(alpha_t.shape) < len(model_output.shape): # For broadcasting
            alpha_t = alpha_t.unsqueeze(-1)
            beta_t  = beta_t.unsqueeze(-1)

        # alpha_t, beta_t is used to model_output and sample (B, ...). We need to global multiply the alpha_t and beta_t to the model_output and sample.

        if self.predict_type == "epsilon":
            # 모델이 노이즈를 직접 예측하므로 pred_epsilon은 pred와 동일
            pred_x_start = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
            pred_epsilon = model_output
        elif self.predict_type == "v_prediction":
            # pred_x_start와 pred_epsilon을 계산
            pred_x_start = alpha_t.sqrt() * sample - beta_t.sqrt() * model_output
            pred_epsilon = alpha_t.sqrt() * model_output + beta_t.sqrt() * sample
        elif self.predict_type == "sample":
            # 모델이 원본 샘플을 직접 예측하므로 pred_x_start는 pred와 동일
            pred_x_start = model_output
            pred_epsilon = (sample - alpha_t.sqrt() * pred_x_start) / beta_t.sqrt()

        return pred_x_start, pred_epsilon


    def forward(self, x_target, x_start, cond=None):
        """
        x_target: torch.Tensor, (B, D).
                 Wheter the diffusion model is conditional or not, the `x_target` data is always ground truth data.
        x_start: torch.Tensor, (B, D).
                 If the diffusion model is conditional, the `x_start` data is the condition data. 
                 But, if the diffusion model is not conditional, the `x_start` data is the same as `noisy_x_target` data.  
        """
        if self.init_x0_scale:
            x_target *= self.scale

        if self.training_clamp_norm: # For empirical results, use `training_clamp_norm` for SEED situation is not recommended.
            x_target = self.norm_clamp_embed(x_target, self.scale)
            x_start  = self.norm_clamp_embed(x_start,  self.scale)

        noise = torch.randn_like(x_target, device=x_target.device) # (B, D)
        batch_size = x_target.shape[0]

        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=x_target.device, dtype=torch.long) # (B,)

        # `SEED` support only self-conditioning DDPM without `cond`. If you want to train SEED-based experiment, set `self_cond` to True.
        input_x = x_target if not self.self_cond else x_start 
        x_t = self.noise_scheduler.add_noise(input_x, noise, timesteps) 

        # Get the target for loss depending on the prediction type
        if self.predict_type == "epsilon":
            target = noise
        elif self.predict_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x_target, noise, timesteps)
        elif self.predict_type == "sample":
            target = x_target
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise and compute loss
        pred = self.model(x_t, timesteps, cond=cond)
        
        """
        TODO: 'training_clamp_norm' 을 diffusers의 DDIMScheduler 과정에서 정확하게 사용되게끔 고려해보기. 아직 정확히 체크안함.
        DDIMScheduler가 predict_type 이 'sample' 일때랑 'noise'일때랑 내부에서 clipping이랑 scaling을 어떻게 하는지 확인해야 함.
        """

        loss = self.loss_fn(pred, target)

        if self.predict_type == "sample" and self.training_clamp_norm:
            pred = self.norm_clamp_embed(pred, self.scale)

        pred_x_target, pred_epsilon = self.step(pred, timesteps, x_t) # DDIMSchedular의 step 함수와 동일한 역할을 함.

        return loss, pred_x_target


    @torch.no_grad()
    def sample(self, x_start, cond=None, sample_timesteps=None, use_ddim=True, self_cond=False, classifier_guidance=False, scale=None, **kwargs):
        """
        x_start : torch.Tensor, (B, D). Different other basic DDPM model, we always use x_start as the conditional input x (self-conditioning).
        cond  : torch.Tensor, (B, D). If use, the `cond` data is the condition data. It is textual guide.
        sample_timesteps : int `T`, default None
        """
        self.noise_scheduler.set_timesteps(sample_timesteps)
        timesteps = self.noise_scheduler.timesteps # shape : (T,), T is the number of sample_timesteps

        noise = torch.randn_like(x_start)

        if scale is None: scale = self.scale
        
        x_start = self.norm_clamp_embed(x_start, scale) # For empirical results, use `norm_clamp_embed` for sampling is better.

        batch_size = x_start.size(0)

        if use_ddim: # use DDIM sampling. For SEED, use_ddim is False because it is not effective than one-step sampling.
            if not self_cond:
                x_t = noise
            else:
                # `SEED` support only self-conditioning DDPM without `cond`. If you want to train SEED-based experiment, set `self_cond` to True.
                T_batch = torch.ones(batch_size, device=x_start.device, dtype=torch.long) * timesteps[0] # (B,)
                x_t = self.noise_scheduler.add_noise(x_start, noise, T_batch) 

            for i, t in enumerate(timesteps):
                #x_t = self.noise_scheduler.scale_model_input(x_t, t) # scale the input, not implemented yet in official DDIMSchedular
                t_batch = torch.ones(batch_size, device=x_t.device, dtype=torch.long) * t

                pred = self.model(x_t, t_batch, cond=cond)

                if self.sampling_clamp_norm and self.predict_type == "sample":
                    pred = self.norm_clamp_embed(pred, scale)

                # compute the previous noise sample x_t -> x_t-1
                x_t = self.noise_scheduler.step(pred, t, x_t, **kwargs).prev_sample

            pred_x_start = x_t

        
        else: # just predict output (noise or x_start) from the t step's instance x_t, directly. So, in here, it works same as self_cond == True situation.
            """ 
            One-step sampling 
            For SEED, we use one-step sampling officially, because it is more faster and effective than traditional DDIM sampling.
            We recommend to use one-step sampling for SEED-based experiment. But, you need to search the best 't' step for appropriate noise magnitude.
            """
            t_batch = torch.ones(batch_size, device=x_start.device, dtype=torch.long) * sample_timesteps # (B,)
            x_t = self.noise_scheduler.add_noise(x_start, noise, t_batch) if self_cond else noise

            pred = self.model(x_t, t_batch, cond=cond)

            if self.sampling_clamp_norm and self.predict_type == "sample":
                pred = self.norm_clamp_embed(pred, scale)

            pred_x_start = self.noise_scheduler.step(pred, sample_timesteps, x_t).pred_original_sample

        # Final normalization and scaling, default is just normalize without scaling.
        if self.sampling_final_norm:
            pred_x_start = self.norm_clamp_embed(pred_x_start)

        return pred_x_start

