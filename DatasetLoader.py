#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0).astype(float) # shape: (num_eval, max_audio)


    return feat;
    

class AugmentWAV(object):
    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[4,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio, noise_snr=None):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = noise_snr if noise_snr is not None else random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):
        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]


class train_dataset_loader(Dataset):
    def __init__(self, train_list, augment, augment_8k, musan_path, rir_path, max_frames, train_path, **kwargs):

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)

        self.train_list = train_list
        self.train_path = train_path
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment
        self.augment_8k = augment_8k
        
        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines();

        random.shuffle(lines)

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_list  = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split();
            dummy_speaker_id, file_path = data[0], data[1]

            filename = os.path.join(train_path, file_path) if self.train_path is not None else file_path 
            """ Data file list policy:
            -> self.train_path is main dataset directory path. If your file_path is absolute path for audio date, you should set self.train_path = None.
            # E.g., file_path = 'rel_path/to/audio_file.wav', train_path = 'abs_path/to/dataset/' --> filename = train_path + file_path = 'abs_path/to/dataset/rel_path/to/audio_file.wav'
            # E.g., file_path = 'abs_path/to/audio_file.wav', train_path = None (must be None)    --> filename = file_path = 'abs_path/to/audio_file.wav'
            """

            self.data_list.append(filename)

    def __getitem__(self, index): 

        filename   = self.data_list[index]
        audio = loadWAV(filename, self.max_frames, evalmode=False)

        audio_music  = self.augment_wav.additive_noise('music',  audio)
        #audio_speech = self.augment_wav.additive_noise('speech', audio) # we dont know that this is necessary for SEED task (Because, speech-ovelap scenario is not considered)
        audio_noise  = self.augment_wav.additive_noise('noise',  audio)
        audio_rir    = self.augment_wav.reverberate(audio)

        feat = [audio, audio, audio_music, audio_noise, audio_rir]

        feat = numpy.concatenate(feat, axis=0)

        return torch.FloatTensor(feat)

    def __len__(self):
        return len(self.data_list)


class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, musan_path, rir_path, **kwargs):
        self.max_frames = eval_frames;
        self.num_eval   = num_eval
        self.test_path  = test_path
        self.test_list  = test_list

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = self.max_frames)

    def __getitem__(self, index):
        filename = os.path.join(self.test_path, self.test_list[index]) if self.test_path is not None else self.test_list[index]
        # Same as train data file policy. Check example above line 136.

        audio = loadWAV(filename, self.max_frames, evalmode=True, num_eval=self.num_eval)

        return torch.FloatTensor(audio), self.test_list[index]
    def __len__(self):
        return len(self.test_list)