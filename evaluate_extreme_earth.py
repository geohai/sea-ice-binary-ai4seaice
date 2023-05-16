import argparse
import configparser
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
import torch
import xarray as xr
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             jaccard_score)
from torch import nn
from torchvision.transforms import Normalize


def evaluate(config):

    dir_out = os.path.normpath(config['io']['dir_out'])
    dset = config['io']['dset']
    
    test_rasters = [os.path.normpath(f) for f in config['io']['test_rasters'].split('\n')]
    test_label_rasters = [os.path.normpath(f) for f in config['io']['test_label_rasters'].split('\n')]

    if 'ensemble' in config['io']:
        ensemble = config['io']['ensemble'] == 'True'

    if 'activate_dropout' in config['io']:
        activate_dropout = config['io']['activate_dropout'] == 'True'

    if 'num_realizations_per_model' in config['io']:
        num_realizations_per_model = int(config['io']['num_realizations_per_model'])
    else:
        num_realizations_per_model = 1
    
    if ensemble:
        model_paths = [os.path.join(f) for f in config['io']['model_path'].split('\n')]
        models = []
        for model_path in model_paths:
            models.append(torch.jit.load(model_path))
        for model in models:
            model.eval()

            if activate_dropout:
                for m in model.decoder.modules():
                    if m.original_name =='Dropout':
                        print('Activating Dropout')
                        m.train()

    else:
        model_path = os.path.normpath(config['io']['model_path'])
        model = torch.jit.load(model_path)
        model.eval()

        if activate_dropout:
            for m in model.decoder.modules():
                if m.original_name =='Dropout':
                    print('Activating Dropout')
                    m.train()

    ignore_index = int(config['datamodule']['ignore_index'])

    label = config['datamodule']['label']
    input_features = ['nersc_sar_primary',
                      'nersc_sar_secondary',
                      'sar_incidenceangle']
    
    global_meanstd = np.load(os.path.join('misc', 'global_meanstd.npy'), allow_pickle=True).item()
    
    mean = [global_meanstd[val]['mean'] for val in input_features]
    std =  [global_meanstd[val]['std'] for val in input_features]
    norms = Normalize(mean, std)

    # save configuration file:
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    with open(os.path.join(dir_out, f'evaluate-{dset}.cfg'), 'w') as out_file:
        config.write(out_file)

    # start dictionary for summary results:
    dict_metrics = {}
    # run on test rasters:
    softmax = nn.Softmax(0)
    for idx, (test_input, test_label) in enumerate(zip(test_rasters, test_label_rasters)):

        print(f'Using raster {test_input}...', end=' ')

        raster_y = rioxarray.open_rasterio(test_label, masked=True)
        y = np.squeeze(raster_y.values, 0)
        y[y==ignore_index]=np.nan
        mask_y = np.isnan(y)

        start_time = time.perf_counter()
                
        raster = rioxarray.open_rasterio(test_input, masked=True)
        x = torch.from_numpy(raster.values).unsqueeze(dim=0)

        # get input mask 
        mask_x = np.isnan(raster.values).any(axis=0)
        
        # get "full" mask
        mask = np.logical_or(mask_x, mask_y)

        # normalize
        x = torch.nan_to_num(norms(x))

        if ensemble:
            res = []
            for model in models:
                for _ in range(num_realizations_per_model):
                    res.append(softmax(torch.squeeze(model(x).detach(),0)))

        else:
            res = []
            with torch.no_grad():
                for _ in range(num_realizations_per_model):
                    res.append(softmax(torch.squeeze(model(x).detach(),0)))
        
        end_time = time.perf_counter()
        print(f'{(end_time-start_time)/60:.2f} minutes for model prediction...', end=' ')
        start_time = time.perf_counter()

        # calculate mean and std:
        res_mean =  torch.mean(torch.stack(res), dim=0)
        res_std =  torch.std(torch.stack(res), dim=0)
        # calculate entropy:
        res_entropy = np.mean(entropy(torch.stack(res, axis=1).cpu().numpy(), 
                                      base=2, axis=0), 
                              axis=0)

        # mark nan vals
        for band in res_mean:
            band[mask_x] = np.nan
        for band in res_std:
            band[mask_x] = np.nan

        res_entropy[mask_x] = np.nan
        res_entropy = np.expand_dims(res_entropy, 0)

        # use raster information to populate output:
        ###### mean
        xr_res = xr.DataArray(res_mean, 
                              [('band', np.arange(1, res_mean.shape[0]+1)),
                              ('y', raster.y.values),
                              ('x', raster.x.values)])
        
        xr_res['spatial_ref']=raster.spatial_ref                              
        xr_res.attrs=raster.attrs
        
        # write to file
        out_fname = os.path.join(dir_out, f'pred-mean-{Path(test_input).stem}.tif')
        if os.path.isfile(out_fname):
            os.remove(out_fname)
        xr_res.rio.to_raster(out_fname, dtype="float32")

        ###### std
        xr_res = xr.DataArray(res_std, 
                              [('band', np.arange(1, res_mean.shape[0]+1)),
                              ('y', raster.y.values),
                              ('x', raster.x.values)])
        
        xr_res['spatial_ref']=raster.spatial_ref                              
        xr_res.attrs=raster.attrs
        
        # write to file
        out_fname = os.path.join(dir_out, f'pred-var-{Path(test_input).stem}.tif')
        if os.path.isfile(out_fname):
            os.remove(out_fname)
        xr_res.rio.to_raster(out_fname, dtype="float32")

        ###### entropy
        xr_res = xr.DataArray(res_entropy, 
                              [('band', [1]),
                              ('y', raster.y.values),
                              ('x', raster.x.values)])
        
        xr_res['spatial_ref']=raster.spatial_ref                              
        xr_res.attrs=raster.attrs
        
        # write to file
        out_fname = os.path.join(dir_out, f'pred-entropy-{Path(test_input).stem}.tif')
        if os.path.isfile(out_fname):
            os.remove(out_fname)
        xr_res.rio.to_raster(out_fname, dtype="float32")

        ##### write the class
        y_pred_class = res_mean.argmax(0)
        # 241 is the no data value for uint8
        nodata = 241
        y_pred_class[mask] = nodata
        y_pred_class = np.expand_dims(y_pred_class, 0)

        xr_res = xr.DataArray(y_pred_class, 
                              [('band', [1]),
                              ('y', raster.y.values),
                              ('x', raster.x.values)])
        
        xr_res['spatial_ref']=raster.spatial_ref                              
        xr_res.attrs=raster.attrs

        xr_res.rio.write_nodata(nodata, inplace=True)
        
        out_fname = os.path.join(dir_out, f'class-{Path(test_input).stem}.tif')
        if os.path.isfile(out_fname):
            os.remove(out_fname)
        xr_res.rio.to_raster(out_fname, dtype="uint8")


        corrects = y == y_pred_class
        corrects = corrects.astype('uint8')
        corrects[np.expand_dims(mask, 0)] = nodata

        xr_res = xr.DataArray(corrects, 
                              [('band', [1]),
                              ('y', raster.y.values),
                              ('x', raster.x.values)])
        
        xr_res['spatial_ref']=raster.spatial_ref                              
        xr_res.attrs=raster.attrs

        xr_res.rio.write_nodata(nodata, inplace=True)
        
        out_fname = os.path.join(dir_out, f'corrects-{Path(test_input).stem}.tif')
        if os.path.isfile(out_fname):
            os.remove(out_fname)
        xr_res.rio.to_raster(out_fname, dtype="uint8")

        #########################################################
        # metrics
        #########################################################
                
        y_true = y[~mask].ravel().astype(int)
        y_pred_class = y_pred_class[~np.expand_dims(mask, 0)].ravel()

        dict_metrics[Path(test_input).stem]={}
        dict_metrics[Path(test_input).stem]['accuracy'] = accuracy_score(y_true, y_pred_class)

        with open(os.path.join(dir_out, f'{dset}-metrics.txt'), 'a', encoding='utf-8') as outfile:

            outfile.write(f'{Path(test_input).stem} performance \n')
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
                
                dict_metrics[Path(test_input).stem][f'iou-{avg}'] = iou
                dict_metrics[Path(test_input).stem][f'f1-{avg}'] =f1            
                
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
            fig.savefig(os.path.join(dir_out, f'{dset}-confusion_matrix-{Path(test_input).stem}.pdf'), bbox_inches='tight')
            plt.close("all")

        end_time = time.perf_counter()
        print(f'{(end_time-start_time)/60:.2f} minutes for writing figures and metrics')

    # save dataframe with metrics
    pd.DataFrame(dict_metrics).T.to_csv(os.path.join(dir_out, f'{dset}-summary_metrics.csv'))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='config_evalEE.ini')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        evaluate(config)
    
    else:
        print('Please provide a valid configuration file.')
