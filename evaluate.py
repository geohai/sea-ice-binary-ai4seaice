import argparse
import configparser
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             jaccard_score)
from torch import nn


def save_fig(scene, lab, dir_out, dset, fname):

    BINARY_GROUPS = {0: 'Water', 
                     1: 'Ice'}
    BINARY_LEVELS = [-0.5,0.5,1.5]
    BINARY_COLORS = ['#67a9cf', '#ef8a62']

    ERROR_GROUPS = {0: 'Mismatch', 
                    1: 'Match'}

    ERROR_LEVELS = [-0.5,0.5,1.5]
    ERROR_COLORS = ['#7570b3', '#1b9e77']

    fill_key = 'chart_fill_value' if 'chart_fill_value' in scene[lab].attrs.keys() else 'variable_fill_value'
    fig, ax = plt.subplots(figsize=(4,4))
    # Locating current axes
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)

    scene['to_plot'].values = scene[lab].values.copy().astype(float)
    scene['to_plot'].values[scene[lab] == scene[lab].attrs[fill_key]] = np.nan

    if ('sar' in lab) or ('std' in lab):
        scene['to_plot'].plot(ax=ax, cbar_ax=cbar_ax, robust=True, cmap='viridis')
    elif 'mean' in lab:
        scene['to_plot'].plot(ax=ax, cbar_ax=cbar_ax, vmin=0.2, vmax=1.0, cmap='viridis')
    elif ('entropy' in lab):
        # vmin ~ 0.95 probability
        scene['to_plot'].plot(ax=ax, cbar_ax=cbar_ax, vmin=0.29, vmax=1., cmap='inferno')
    elif any(cc in lab for cc in ['binary', 'corrects']):
        if 'corrects' in lab:
            BINARY_GROUPS = ERROR_GROUPS
            BINARY_LEVELS = ERROR_LEVELS
            BINARY_COLORS = ERROR_COLORS
        scene['to_plot'].plot(ax=ax, cbar_ax=cbar_ax, 
                              levels=BINARY_LEVELS, 
                              colors=BINARY_COLORS)
        cbar_ax.set_yticks(list(BINARY_GROUPS.keys()))                              
        cbar_ax.set_yticklabels(list(BINARY_GROUPS.values()), rotation=90)

    cbar_ax.set_ylabel('')
    ax.set_aspect('equal')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    fig.savefig(os.path.join(dir_out, f'{dset}-{Path(fname).stem}-{lab}.png'), dpi=600, bbox_inches='tight')
    plt.close('all')

def evaluate(config):

    dir_out = os.path.normpath(config['io']['dir_out'])
    dir_in = os.path.normpath(config['io']['dir_in'])
    dset = config['io']['dset']
    
    if 'file_list' in config['io']:
        test_files = [os.path.join(dir_in, f) for f in config['io']['file_list'].split('\n')]
    else:
        test_files = [os.path.join(dir_in, f) for f in os.listdir(dir_in) if f.endswith('.nc')]

    if 'ensemble' in config['io']:
        ensemble = config['io']['ensemble'] == 'True'
    
    if ensemble:
        model_paths = [os.path.normpath(f) for f in config['io']['model_path'].split('\n')]
        models = []
        for model_path in model_paths:
            models.append(torch.jit.load(model_path))
        for model in models:
            model.eval()
    else:
        model_path = os.path.normpath(config['io']['model_path'])
        model = torch.jit.load(model_path)
        model.eval()

    label = config['datamodule']['label']
    input_features = ['nersc_sar_primary',
                      'nersc_sar_secondary',
                      'sar_incidenceangle']
    
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    # save configuration file:
    with open(os.path.join(dir_out, f'evaluate-{dset}.cfg'), 'w') as out_file:
        config.write(out_file)

    # start dictionary for summary results:
    dict_metrics = {}
    # run on test rasters:
    softmax = nn.Softmax(0)
    for idx, test_file in enumerate(test_files):

        print(f'Using raster {test_file}...', end=' ')
        start_time = time.perf_counter()
        
        scene = xr.open_dataset(test_file)

        if label == 'binary':
            scene['binary'] = scene['SIC'].copy()
            scene['binary'].values[scene['binary'].values > 1] = 1
            scene['binary'].values[scene['SIC'].values == scene['SIC'].chart_fill_value] = scene['SIC'].chart_fill_value

        x = torch.from_numpy(np.concatenate([np.expand_dims(scene[val], axis=0) for val in input_features], axis=0)).type(torch.float)
        x = x.unsqueeze(axis=0)
        y = scene[label].values
        
        if ensemble:
            res = []
            res_std = []
            for model in models:
                res.append(softmax(torch.squeeze(model(x).detach(),0)))
           
            res_mean = torch.mean(torch.stack(res), dim=0).detach().numpy()
            res_std =  torch.std(torch.stack(res), dim=0).detach().numpy()
            # calculate entropy:
            res_entropy = np.mean(entropy(torch.stack(res, axis=1).cpu().numpy(), 
                                        base=2, axis=0), 
                                axis=0)

            res = res_mean

        else:
            with torch.no_grad():
                res = model(x)
                # compute probabilities (instead of scores):
                res = softmax(torch.squeeze(res,0)).detach().numpy()

        end_time = time.perf_counter()
        print(f'{(end_time-start_time)/60:.2f} minutes for model prediction...', end=' ')
        start_time = time.perf_counter()

        # write the class
        y_pred_class = res.argmax(0)

        corrects = y == y_pred_class

        # can't generate results with invalid input:
        mask_pred= scene[input_features[0]].values == scene[input_features[0]].variable_fill_value 
        # compare with valid pixels only:
        mask_target = scene[label].values == scene[label].chart_fill_value

        mask = ~np.logical_or(mask_pred, mask_target)

        # save in scene for plotting
        scene[f'pred-{label}'] = scene[label].copy()
        scene[f'pred-{label}'].values = y_pred_class.copy()
        scene[f'pred-{label}'].values[~mask] = scene[label].chart_fill_value

        scene['corrects'] = scene[label].copy()
        scene['corrects'].values = corrects.copy().astype(int)
        scene['corrects'].values[~mask] = scene[label].chart_fill_value

        if ensemble:
            scene[f'mean-{label}'] = scene['nersc_sar_primary'].copy()
            scene[f'mean-{label}'].values = res_mean[1].copy()
            scene[f'mean-{label}'].values[scene['nersc_sar_primary']==scene['nersc_sar_primary'].variable_fill_value] = scene['nersc_sar_primary'].variable_fill_value

            scene[f'std-{label}'] = scene['nersc_sar_primary'].copy()
            scene[f'std-{label}'].values = res_std[1].copy()
            scene[f'std-{label}'].values[scene['nersc_sar_primary']==scene['nersc_sar_primary'].variable_fill_value] = scene['nersc_sar_primary'].variable_fill_value

            scene[f'entropy-{label}'] = scene['nersc_sar_primary'].copy()
            scene[f'entropy-{label}'].values = res_entropy.copy()
            scene[f'entropy-{label}'].values[scene['nersc_sar_primary']==scene['nersc_sar_primary'].variable_fill_value] = scene['nersc_sar_primary'].variable_fill_value

        #########################################################
        # metrics
        #########################################################
                
        y_true = y[mask].ravel()
        y_pred_class = y_pred_class[mask].ravel()

        dict_metrics[Path(test_file).stem]={}
        dict_metrics[Path(test_file).stem]['accuracy'] = accuracy_score(y_true, y_pred_class)

        with open(os.path.join(dir_out, f'{dset}-metrics.txt'), 'a', encoding='utf-8') as outfile:

            outfile.write(f'{Path(test_file).stem} performance \n')
            outfile.write(classification_report(y_true, 
                                                y_pred_class))
            outfile.write('\n')
            outfile.write(f'Jaccard Index: \n')
            for avg in ['micro', 'macro', 'weighted']:
                iou = jaccard_score(y_true, 
                                    y_pred_class, average=avg)

                outfile.write(f'{avg}: {iou:.2f} \n')

                # compute f1 for dataframe
                f1 = f1_score(y_true, 
                                    y_pred_class, average=avg)
                
                dict_metrics[Path(test_file).stem][f'iou-{avg}'] = iou
                dict_metrics[Path(test_file).stem][f'f1-{avg}'] =f1            
                
            cm = confusion_matrix(y_true, 
                                  y_pred_class, 
                                  normalize='true')
            outfile.write('\n')
            outfile.write(f'Confusion Matrix: \n')
            for row in cm:
                for col in row:
                    outfile.write(f'      {col:.2f}')
                outfile.write('\n')
            outfile.write('\n\n')

            # save pdf 
            fig, ax = plt.subplots(figsize=(3.5,3.5))
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)

            ConfusionMatrixDisplay.from_predictions(y_true, 
                                                    y_pred_class, 
                                                    normalize='true', 
                                                    values_format = '.2f',
                                                    ax=ax,
                                                    colorbar=False, 
                                                    im_kw={'vmin': 0, 'vmax': 1})

            im = ax.images[-1]
            fig.colorbar(im, cax=cbar_ax)    
            cbar_ax.set_ylabel('Proportion')
            fig.tight_layout()
            fig.savefig(os.path.join(dir_out, f'{dset}-confusion_matrix-{Path(test_file).stem}.pdf'), bbox_inches='tight')
            plt.close("all")

        #########################################################
        # images
        #########################################################

        # save target and prediction images
        scene['to_plot'] = scene[input_features[0]].copy()
        for lab in [f'pred-{label}', label, 'corrects']+input_features:
            save_fig(scene, lab, dir_out, dset, test_file)
        if ensemble:
            for lab in [f'mean-{label}', f'std-{label}', f'entropy-{label}']:
                save_fig(scene, lab, dir_out, dset, test_file)

        end_time = time.perf_counter()
        print(f'{(end_time-start_time)/60:.2f} minutes for writing figures and metrics')

    # save dataframe with metrics
    pd.DataFrame(dict_metrics).T.to_csv(os.path.join(dir_out, f'{dset}-summary_metrics.csv'))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='config_eval.ini')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        evaluate(config)
    
    else:
        print('Please provide a valid configuration file.')
