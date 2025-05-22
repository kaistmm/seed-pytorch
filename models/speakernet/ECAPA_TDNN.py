#! /usr/bin/python
# -*- encoding: utf-8 -*-

## Here, log_input forces alternative mfcc implementation with pre-emphasis instead of actual log mfcc

import math
import torch
import torch.nn as nn
import torchaudio
from models.speakernet.utils import PreEmphasis

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 4):

        super(Bottle2neck, self).__init__()

        width       = int(math.floor(planes / scale))
        
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        
        self.nums   = scale -1

        convs       = []
        bns         = []

        num_pad = math.floor(kernel_size/2)*dilation

        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))

        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)

        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)

        self.relu   = nn.ReLU()

        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)

        out += residual

        return out 

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class ECAPA_TDNN(nn.Module):
    def __init__(self, block, C, model_scale=8, nOut=192, encoder_type='ECA', n_mels=80, context=True, summed=True, log_input=True, specaug=False, **kwargs):
        super(ECAPA_TDNN, self).__init__()

        self.context    = context
        self.summed     = summed
        self.n_mfcc     = n_mels
        self.encoder_type = encoder_type
        self.log_input  = log_input

        self.scale  = model_scale

        self.conv1  = nn.Conv1d(self.n_mfcc, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        
        self.layer1 = block(C, C, kernel_size=3, dilation=2, scale=self.scale)
        self.layer2 = block(C, C, kernel_size=3, dilation=3, scale=self.scale)
        self.layer3 = block(C, C, kernel_size=3, dilation=4, scale=self.scale)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)

        if specaug:
            print("Use SpecAugment into the ECAPA_TDNN model")
        self.specaug = FbankAug() if specaug else False # Spec augmentation 
        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchmfcc      = torch.nn.Sequential(
            PreEmphasis(),
            #torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window, n_mels=n_mels),
            torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc = n_mels, log_mels=self.log_input, dct_type = 2, melkwargs={'n_mels': n_mels, 'n_fft':512, 'win_length':400, 'hop_length':160, 'f_min':20, 'f_max':7600, 'window_fn':torch.hamming_window}),
            )

        if self.context:
            attn_input = 1536*3
        else:
            attn_input = 1536

        if self.encoder_type == 'ECA':
            attn_output = 1536
        elif self.encoder_type == 'ASP':
            attn_output = 1
        else:
            raise ValueError('Undefined encoder')


        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
            )

        self.bn5 = nn.BatchNorm1d(1536*2)

        self.fc6 = nn.Linear(1536*2, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)

        self.nOut = nOut

    def forward_extract_feat(self, x, is_waveform=True, normalize=True):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if is_waveform:
                    x = self.torchmfcc(x) + 1e-6
                    #x = x.log()   
                if normalize:
                    #x = self.instancenorm(x)
                    x = x - torch.mean(x, dim=-1, keepdim=True)
                    
                if self.specaug:
                    x = self.specaug(x)                
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        if self.summed:
            x1 = self.layer1(x)
            x2 = self.layer2(x+x1)
            x3 = self.layer3(x+x1+x2)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1)) # [B, D*l, T]
        x = self.relu(x)

        return x

    def forward_extract_embed(self, x):
        t = x.size()[-1]

        if self.context:
            global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        else:
            global_x = x

        w = self.attention(global_x)

        mu  = torch.sum(x * w, dim=2)
        sg  = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4))
        l2  = torch.sqrt( ( torch.sum((x**2) * w, dim=2)).clamp(min=1e-4))

        x = torch.cat((l2,sg),1)
        #x = mu

        x = self.bn5(x)
        x = self.fc6(x)

        return x

    def forward(self, x, is_waveform=True, normalize=True):
        x = self.forward_extract_feat(x, is_waveform, normalize)
        x = self.forward_extract_embed(x)

        return x


def MainModel(**kwargs):
    model = ECAPA_TDNN(Bottle2neck, context=True, summed=True,  **kwargs)
    return model