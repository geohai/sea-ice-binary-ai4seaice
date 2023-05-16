#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr
from numpy.random import MT19937, RandomState, SeedSequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

class SARBinaryAI4ArcticChallengeDataset(Dataset):
    """Pytorch dataset for loading batches of patches of scenes from the ASID V2 data set."""

    def __init__(self, options, files):
        self.options = options
        self.files = files

        # create a BitGenerator for numpy sampling
        self.rs = RandomState(MT19937(SeedSequence(options['seed'])))

    def __len__(self):
        return len(self.samples_dict)

    def prepare_samples(self):
        """Creates cropped samples to be used for training and validation.
        """

        # reset samples
        self.samples_dict = {}
        # maximum number of attempts to extract a patch from a scene
        max_attempts = 100
        # label:
        lab = self.options['label']

        print(f'Preparing {self.options["dset"]} samples...')

        if self.options['dset'] == 'validation':

            # save list of files
            self.samples_dict = {key: val for key, val in enumerate(self.files)}

        elif self.options['dset'] == 'train':
            
            for full_attempts in range(max_attempts):
    
                # select random scenes: 
                scenes = self.rs.choice(self.files, self.options['num_samples']-len(self.samples_dict))

                for scene_name in tqdm(scenes):
                    scene = xr.open_dataset(os.path.join(self.options['dir_in'], scene_name))

                    if lab == 'binary':
                        scene['binary'] = scene['SIC'].copy()
                        scene['binary'].values[scene['binary'].values > 1] = 1
                        scene['binary'].values[scene['SIC'].values == scene['SIC'].chart_fill_value] = scene['SIC'].chart_fill_value

                    for attempts in range(max_attempts):
                        # randomly select locations
                        row_rand = np.random.randint(low=0, high=scene['SIC'].values.shape[0] - self.options['patch_size'])
                        col_rand = np.random.randint(low=0, high=scene['SIC'].values.shape[1] - self.options['patch_size'])

                        # keep if at least 30% of pixels are valid
                        samp_y = scene['SIC'].isel(
                                        sar_lines=range(row_rand, row_rand + self.options['patch_size']),
                                        sar_samples=range(col_rand, col_rand + self.options['patch_size'])).values

                        if np.sum(samp_y == scene['SIC'].chart_fill_value)/samp_y.ravel().shape[0] < .30:
                            # save sample information in the full sample_dict
                            # dictionary to save sample information
                            sample_dict = {} 
                            
                            for source_name in ['nersc_sar_primary',                                       
                                                'nersc_sar_secondary',
                                                'sar_incidenceangle',
                                                lab]:

                                sample_dict[source_name] = scene[source_name].isel(
                                        sar_lines=range(row_rand, row_rand + self.options['patch_size']),
                                        sar_samples=range(col_rand, col_rand + self.options['patch_size'])).values

                            self.samples_dict[len(self.samples_dict)] = sample_dict.copy()
                            
                            break
                    
                        if attempts == max_attempts-1:
                            if self.options['verbose']:
                                print(f'--I could not find valid samples for {scene_name}')
                                
                    scene = None

                if len(self.samples_dict) == self.options['num_samples']:
                    break
                else:
                    if self.options['verbose']:
                        print(f'----I could not select {self.options["num_samples"]} samples in {full_attempts+1} attempts.')            
               
                if full_attempts == max_attempts-1:
                    if self.options['verbose']:
                        print(f'----Warning! I could not complete the training set')


    def __len__(self):
        return self.options['epoch_len']

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.options['dset'] == 'validation':

            scene = xr.open_dataset(os.path.join(self.options['dir_in'], self.samples_dict[idx]))

            if self.options['label'] == 'binary':
                scene['binary'] = scene['SIC'].copy()
                scene['binary'].values[scene['binary'].values > 1] = 1
                scene['binary'].values[scene['SIC'].values == scene['SIC'].chart_fill_value] = scene['SIC'].chart_fill_value

            # dictionary to save sample information
            sample_dict = {} 
            for source_name in [ 'nersc_sar_primary',                                       
                                 'nersc_sar_secondary',
                                 'sar_incidenceangle',
                                 self.options['label']]:

                sample_dict[source_name] = scene[source_name].values

            scene = None

            x = torch.from_numpy(np.concatenate([np.expand_dims(sample_dict[val], axis=0) for val in ['nersc_sar_primary',
                                                                                                    'nersc_sar_secondary',
                                                                                                    'sar_incidenceangle',]], axis=0)).type(torch.float)
            y = torch.from_numpy(sample_dict[self.options['label']]).type(torch.long)
            
        else:
            # the dataset contains options['num_samples'] samples, 
            # but idx varies from 0 to __len__
            # so we use idx to randomly select a sample from the sample dictionary
            if self.__len__() < self.options['epoch_len']:
                rand_idx = np.random.choice(np.arange(0, len(self.samples_dict)), size=self.__len__())
                rand_idx = rand_idx[idx]
            else:
                rand_idx = idx
            sample_dict = self.samples_dict[rand_idx]

            x = torch.from_numpy(np.concatenate([np.expand_dims(sample_dict[val], axis=0) for val in ['nersc_sar_primary',
                                                                                                    'nersc_sar_secondary',
                                                                                                    'sar_incidenceangle',]], axis=0)).type(torch.float)
            y = torch.from_numpy(sample_dict[self.options['label']]).type(torch.long)

        
        return x, y

#############################################
#############################################

class GeoDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # the dictionary might be modified when model is trained with multiple workers/gpu
        # save training len:
        self.hparams['epoch_len_training'] = self.hparams['epoch_len']

    def setup(self, stage):

        if stage == 'fit':
            
            self.hparams['dset'] = 'train'
            # make sure 'epoch_len' is restored:
            self.hparams['epoch_len'] = self.hparams['epoch_len_training']
            self.train_ds = SARBinaryAI4ArcticChallengeDataset(files=self.hparams.train_files, options=self.hparams.copy())

            self.hparams['dset'] = 'validation'
            self.hparams['epoch_len'] = len(self.hparams['val_files'])
            self.val_ds = SARBinaryAI4ArcticChallengeDataset( files=self.hparams.val_files, options=self.hparams.copy())

        print(f'{stage} setup complete.')
    
    def train_dataloader(self):
        if self.hparams.verbose:
            print('\nSetting train dataloader')

        if self.hparams['load_samples_dict']:
            with open(os.path.join(self.hparams['dir_out'], 'train_dict.pkl'), 'rb') as fin:
                self.train_ds.samples_dict = pickle.load(fin)
        else:
            self.train_ds.prepare_samples()

            if self.hparams['save_samples_dict']:
                # save dictionaries:
                with open(os.path.join(self.hparams['dir_out'], 'train_dict.pkl'), 'wb') as fout:
                    pickle.dump(self.train_ds.samples_dict, fout)

        if self.hparams.verbose:
            print('A sample of train: ', end='')
            x, y = self.train_ds[0]
            print(f'has shape x = {x.shape} y = {y.shape}')

        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        if self.hparams.verbose:
            print('\nSetting val dataloader')

        if self.hparams['load_samples_dict']:
            with open(os.path.join(self.hparams['dir_out'], 'val_dict.pkl'), 'rb') as fin:
                self.val_ds.samples_dict = pickle.load(fin)
        else:
            self.val_ds.prepare_samples()

            if self.hparams['save_samples_dict']:
                # save dictionaries:
                with open(os.path.join(self.hparams['dir_out'], 'val_dict.pkl'), 'wb') as fout:
                    pickle.dump(self.val_ds.samples_dict, fout)

        if self.hparams.verbose:
            print('A sample of validation: ', end='')
            x, y = self.val_ds[0]
            print(f'has shape x = {x.shape} y = {y.shape}')

        return DataLoader(self.val_ds, batch_size=1)
