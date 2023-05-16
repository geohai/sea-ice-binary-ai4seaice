import argparse
import configparser
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from loaders import GeoDataModule
from models import ResNetASPP


def main(config):

    dir_out = os.path.normpath(config['io']['dir_out'])
    dir_in = os.path.normpath(config['io']['dir_in'])
    files_to_ignore = [val for val in config['io']['files_to_ignore'].split('\n')]

    num_classes = int(config['model']['num_classes'])
    pretrained = config['model']['pretrained'] == 'True'
    frozen_start = config['model']['frozen_start'] == 'True'
    
    loss = config['loss']['loss']
    if loss == 'focal':
        gamma = float(config['loss']['gamma'])
        alpha = float(config['loss']['alpha'])
    else:
        gamma = None
        alpha = None

    label = config['datamodule']['label']
    verbose = config['datamodule']['verbose'] == 'True'
    epoch_len = int(config['datamodule']['epoch_len'])
    num_samples = int(config['datamodule']['num_samples'])
    load_samples_dict = config['datamodule']['load_samples_dict'] == 'True'
    save_samples_dict = config['datamodule']['save_samples_dict'] == 'True'
    patch_size = int(config['datamodule']['patch_size'])
    num_val_scenes = int(config['datamodule']['num_val_scenes'])
    seed = int(config['datamodule']['seed'])

    min_epochs = int(config['train']['min_epochs'])
    max_epochs = int(config['train']['max_epochs'])
    patience = int(config['train']['patience'])
    reduce_lr_patience = int(config['train']['reduce_lr_patience'])
    batch_size = int(config['train']['batch_size'])
    lr = float(config['train']['lr'])
    reload_every_n_epochs = int(config['train']['reload_every_n_epochs'])

    fine_tune = config['train']['fine_tune'] == 'True'
    ignore_index = int(config['train']['ignore_index'])

    pl.seed_everything(seed, workers=True)

    #########################################################
    # Set training and validation lists.
    #########################################################

    train_files = [f for f in os.listdir(dir_in) if f.endswith('.nc')]
    # Remove ignored files from the train list.
    train_files = [scene for scene in train_files if scene not in files_to_ignore]
    
    # Select a random number of validation scenes with the same seed
    np.random.seed(seed)

    val_files = np.random.choice(np.array(train_files), size=num_val_scenes , replace=False)
    # Remove the validation scenes from the train list.
    train_files = [scene for scene in train_files if scene not in val_files]
    print(f'Training with {len(train_files)} scenes.')

    # Windows need different distributed backend
    # this will be depecrated in pytorch lightning in version 1.8
    # and a full Strategy object will have to be set up
    if os.name == 'nt':
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

    ###########################################################
    # Set the pl.DataModule to be used in experiments
    ###########################################################

    # create datamodule:
    dm = GeoDataModule(train_files=train_files, val_files = val_files,
                       dir_in = dir_in, 
                       dir_out=dir_out,
                       load_samples_dict=load_samples_dict,
                       save_samples_dict=save_samples_dict,
                       label = label, 
                       verbose=verbose, 
                       epoch_len=epoch_len,
                       num_samples=num_samples,
                       patch_size=patch_size,
                       seed=seed, 
                       batch_size=batch_size, 
                    )
    dm.setup('fit')

    ###########################################################
    # create models
    ###########################################################

    model = ResNetASPP( num_classes=num_classes,
                        pretrained=pretrained,
                        frozen_start=frozen_start,
                        loss=loss,
                        gamma=gamma,
                        alpha=alpha,
                        lr=lr, 
                        reduce_lr_patience=reduce_lr_patience, 
                        ignore_index=ignore_index)

    if fine_tune:
        if os.path.isfile(os.path.normpath(config['io']['fname_model'])):
            model.load_from_checkpoint(os.path.normpath(config['io']['fname_model']))
        else:
            print(f"{os.path.normpath(config['io']['fname_model'])} is not a valid model")
            return

    ###########################################################
    # run experiment
    ###########################################################

    # callbacks:
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=patience,
        verbose=False,
        mode='min'
    )

    best_weights = ModelCheckpoint(dirpath=dir_out,
                                    filename=f'best_weights',
                                    save_top_k=1,
                                    verbose=False,
                                    monitor='val_loss',
                                    mode='min'
                                    )
    

    # loggers:
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=dir_out)
    cvs_logger = pl_loggers.CSVLogger(save_dir=dir_out)

    # make sure there are no old files there
    if os.path.isfile(os.path.join(dir_out, 'best_weights.ckpt')):
        os.remove(os.path.join(dir_out, 'best_weights.ckpt'))

    trainer = pl.Trainer(gpus=-1,
                        default_root_dir=dir_out,
                        gradient_clip_val=1.0,   # clip large gradients
                        log_every_n_steps=1,
                        min_epochs=min_epochs,
                        callbacks=[early_stopping, best_weights],
                        logger=[tb_logger, cvs_logger],
                        strategy="ddp",
                        accelerator="gpu",
                        max_epochs=max_epochs,
                        reload_dataloaders_every_n_epochs=reload_every_n_epochs,
                        )
    
    trainer.fit(model, dm)

    # TODO: maybe move what is possible to move
    # pytorch lightning "does not consider" script continues after .fit, 
    # https://stackoverflow.com/questions/66261729/pytorch-lightning-duplicates-main-script-in-ddp-mode
    if model.global_rank != 0:
        sys.exit(0)

    # make sure model has the best weights and not the ones for the last epoch
    if os.path.isfile(os.path.join(dir_out, 'best_weights.ckpt')):
        model.load_from_checkpoint(os.path.join(dir_out, 'best_weights.ckpt'))

    # save for use in production environment
    script = model.to_torchscript()
    torch.jit.save(script, os.path.join(dir_out, "model.pt"))

    # save configuration file:
    with open(os.path.join(dir_out, 'train.cfg'), 'w') as out_file:
        config.write(out_file)
    
    # save configuration file for evaluation of validation samples
    config = configparser.ConfigParser()
    config['io'] = {'dir_out': dir_out,
                    'model_path': os.path.join(dir_out, "model.pt"),
                    'dir_in': dir_in, 
                    'file_list': '\n '.join(val_files),
                    'dset': 'validation'
                   }
    config['datamodule'] = {'label': label}

    with open(os.path.join(dir_out, 'eval_val.cfg'), 'w') as out_file:
        config.write(out_file)

    print(f'Program terminated without errors.')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='config_main.ini')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        main(config)
    
    else:
        print('Please provide a valid configuration file.')