import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import rioxarray
import torch
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from torch import nn

dir_out = os.path.normpath('E:/rafael/data/AI4Arctic/results/v1')
dpi = 500
file_format = 'png'

def _plot_input_rgb(scene, fname, fignumber):
    input_features = ['nersc_sar_primary',
                    'nersc_sar_secondary',
                    'sar_incidenceangle']
    
    # mark nans
    for var in input_features:
        scene[var].values[scene[var] == scene[var].attrs['variable_fill_value']] = np.nan 

    # clip 2-98% and rescale
    scaler = MinMaxScaler(clip=True)
    to_plot = scene[input_features].to_array().data
    for idx, band in enumerate(to_plot):
        clip_min, clip_max = np.nanpercentile(band, [2,98])
        to_plot[idx] = scaler.fit_transform(np.clip(band, clip_min, clip_max).reshape(-1,1)).reshape(band.shape)

    # swap axis to plot rgb and rotate to match other plots
    to_plot = np.swapaxes(to_plot, 0, -1)
    to_plot = np.rot90(to_plot)

    # change nan to 1 so it shows up as white
    to_plot = np.nan_to_num(to_plot, nan=1.0)
    
    fig, ax = plt.subplots(figsize=(4.4,3.8))
    ax.imshow(to_plot)

    ax.set_aspect('equal')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    fig.savefig(os.path.join(dir_out, f'fig{fignumber}-{Path(fname).stem}-rgb.{file_format}'), dpi=dpi)
    plt.close('all')

def _crop_EE(da, east, north, crop_len):
    # use masks
    mask_x = (da.x >= east-crop_len/2) & (da.x <= east+crop_len/2)
    mask_y = (da.y >= north-crop_len/2) & (da.y <= north+crop_len/2)

    # save crop
    da = da[dict(x=mask_x, y=mask_y)]
    return da

def _plot_EE_rgb(fname, fignumber, east=None, north=None, crop_len=None):
    
    da = rioxarray.open_rasterio(fname, masked=True)
    dlabel, cmap_label = _get_ice_mask(fname, da)

    if east and north:
        # use masks
        da = _crop_EE(da, east, north, crop_len)
        dlabel = _crop_EE(dlabel, east, north, crop_len)

    # clip 2-98% and rescale
    scaler = MinMaxScaler(clip=True)

    for idx, band in enumerate(da.data):
        clip_min, clip_max = np.nanpercentile(band, [2,98])
        da.data[idx] = scaler.fit_transform(np.clip(band, clip_min, clip_max).reshape(-1,1)).reshape(band.shape)
    fig, ax = plt.subplots(figsize=(4.4,3.8))
    # Locating current axes
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)

    da.plot.imshow(ax=ax)

    dlabel[0].plot.contour(ax=ax, 
                    cmap=cmap_label,
                    vmax=1,
                    linewidths=0.6, 
                    linestyles = 'dotted'
                    )


    cbar_ax.set_ylabel('')
    cbar_ax.axis('off')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('')
    fig.tight_layout()

    fig.savefig(os.path.join(dir_out, f'fig{fignumber}-{Path(fname).stem}.{file_format}'), dpi=dpi)
    plt.close('all')

def _plot_input(scene, fname, fignumber):
    input_features = ['nersc_sar_primary',
                    'nersc_sar_secondary',
                    'sar_incidenceangle']
    
    for lab in input_features:
        fill_key = 'chart_fill_value' if 'chart_fill_value' in scene[lab].attrs.keys() else 'variable_fill_value'

        fig, ax = plt.subplots(figsize=(4.4,3.8))
        # Locating current axes
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right",
                                        size="10%",
                                        pad=0.1)
        scene['to_plot'] = scene[lab].copy()
        scene['to_plot'].values = scene[lab].values.copy().astype(float)
        scene['to_plot'].values[scene[lab] == scene[lab].attrs[fill_key]] = np.nan
        scene['to_plot'].plot(ax=ax, cbar_ax=cbar_ax, robust=True, cmap='viridis')

        cbar_ax.set_ylabel('')
        ax.set_aspect('equal')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        fig.savefig(os.path.join(dir_out, f'fig{fignumber}-{Path(fname).stem}-{lab}.{file_format}'), dpi=dpi)
        plt.close('all')


def _plot_mean_or_entropy(scene, fname, fignumber, lab='entropy'):
    
    fill_key = 'variable_fill_value'

    fig, ax = plt.subplots(figsize=(4.4,3.8))
    # Locating current axes
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
    scene['to_plot'] = scene[lab].copy()
    scene['to_plot'].values = scene[lab].values.copy().astype(float)
    scene['to_plot'].values[scene[lab] == scene[lab].attrs[fill_key]] = np.nan

    if lab == 'entropy':
        scene['to_plot'].plot(ax=ax, cbar_ax=cbar_ax, vmin=0.29, vmax=1.0, cmap='inferno')
    else:
        scene['to_plot'].plot(ax=ax, cbar_ax=cbar_ax, vmin=0.20, vmax=1.0, cmap='viridis')
    
    cbar_ax.set_ylabel('')
    ax.set_aspect('equal')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    fig.savefig(os.path.join(dir_out, f'fig{fignumber}-{Path(fname).stem}-{lab}.{file_format}'), dpi=dpi)
    plt.close('all')

def _plot_EE_prob_or_entropy(fname, fignumber, east=None, north=None, crop_len=None):

    da = rioxarray.open_rasterio(fname, masked=True)
    dlabel, cmap_label = _get_ice_mask(fname, da)

    if east and north:
        # use masks
        da = _crop_EE(da, east, north, crop_len)
        dlabel = _crop_EE(dlabel, east, north, crop_len)

    fig, ax = plt.subplots(figsize=(4.4,3.8))
    # Locating current axes
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
    if 'mean' in Path(fname).stem:
        da[1].plot.imshow(ax=ax, cbar_ax=cbar_ax, vmin=0.2, vmax=1.0, cmap='viridis')
    elif 'entropy' in Path(fname).stem:
        da[0].plot.imshow(ax=ax, cbar_ax=cbar_ax, cmap='inferno', vmin=0.29, vmax=1.0)
    
    dlabel[0].plot.contour(ax=ax, 
                    cmap=cmap_label,
                    vmax=1,
                    linewidths=0.6, 
                    linestyles = 'dotted'
                    )

    ax.set_title('')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()

    fig.savefig(os.path.join(dir_out, f'fig{fignumber}-{Path(fname).stem}.{file_format}'), dpi=dpi)
    plt.close('all')

def _get_binary_info(lab):
    if lab == 'corrects':
        BINARY_GROUPS = {0: 'Mismatch', 
                        1: 'Match'}
        BINARY_LEVELS = [-0.5,0.5,1.5]
        BINARY_COLORS = ['#7570b3', '#1b9e77']
    else:
        BINARY_GROUPS = {0: 'Water', 
                        1: 'Ice'}
        BINARY_LEVELS = [-0.5,0.5,1.5]
        BINARY_COLORS = ['#67a9cf', '#ef8a62']

    return BINARY_GROUPS, BINARY_LEVELS, BINARY_COLORS

def _get_ice_mask(fname, da):
    root_label = os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/poly_type_wland')
    label_tag = 'poly_type'
    num_classes = 1

    # this file should be associated with a label:
    label_fname = os.path.join(root_label, f'seaice_s1_{Path(fname).stem.split("-")[-1]}-{label_tag}.tif')
    dlabel = rioxarray.open_rasterio(label_fname, masked=True)

    # remove land pixels on th  e input data
    land_mask = np.broadcast_to(dlabel.data == num_classes+1, da.data.shape)
    da.data[land_mask] = np.nan
    
    dlabel.data[dlabel.data > num_classes] = num_classes
    cmap_label = ListedColormap([(255/255, 255/255, 255/255)])

    return dlabel, cmap_label


def _plot_label(scene, fname, fignumber, lab='binary'):

    BINARY_GROUPS, BINARY_LEVELS, BINARY_COLORS = _get_binary_info(lab)

    fill_key = 'chart_fill_value' if 'chart_fill_value' in scene[lab].attrs.keys() else 'variable_fill_value'
    
    fig, ax = plt.subplots(figsize=(4.4,3.8))
    # Locating current axes
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
    
    scene['to_plot'] = scene[lab].copy()


    scene['to_plot'].values = scene[lab].values.copy().astype(float)
    scene['to_plot'].values[scene[lab] == scene[lab].attrs[fill_key]] = np.nan
    scene['to_plot'].plot(ax=ax, cbar_ax=cbar_ax)

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

    fig.savefig(os.path.join(dir_out, f'fig{fignumber}-{Path(fname).stem}-{lab}.{file_format}'), dpi=dpi)
    plt.close('all')

def _plot_EE_icetype(fname, fignumber, east=None, north=None, crop_len=None):

    CLASS_GROUPS = {0: 'NI', 
                    1: 'Nilas', 
                    2: 'YI', 
                    3: 'FYI', 
                    4: 'OI', 
                    5: 'W'} # 6 is land
    CLASS_COLORS = ListedColormap([(253/255, 204/255, 224/255), 
                                    (152/255, 111/255, 196/255), 
                                    (228/255, 0/255, 217/255), 
                                    (250/255, 243/255, 13/255), 
                                    (231/255, 61/255, 4/255), 
                                    (51/255, 153/255, 255/255)
                                    ])

    fig, ax = plt.subplots(figsize=(4.4,3.8))
    # Locating current axes
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
    
    da = rioxarray.open_rasterio(fname, masked=True)
    # replace land with nan:
    da.data[da.data==6] = np.nan

    if east and north:
        # use masks
        da = _crop_EE(da, east, north, crop_len)
    
    da[0].plot.imshow(ax=ax, 
                    vmin=0,
                    vmax=len(CLASS_COLORS.colors),
                    interpolation='none', 
                    cmap=CLASS_COLORS, 
                    cbar_ax=cbar_ax)

    cbar_ax.set_yticks(np.array(list(CLASS_GROUPS.keys()))+0.5)                              
    cbar_ax.set_yticklabels(list(CLASS_GROUPS.values()), rotation=90)

    cbar_ax.set_ylabel('')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('')
    fig.tight_layout()

    fig.savefig(os.path.join(dir_out, f'fig{fignumber}-{Path(fname).stem}.{file_format}'), dpi=dpi)
    plt.close('all')

def _plot_EE_label(fname, fignumber, east=None, north=None, crop_len=None, as_rect=False, lab='binary'):
    """Plots EE labels: binary or errors

    Args:
        fname (str): path to raster file
        fignumber (str): figure number tag
        east (int, optional): east coordinate to crop or locate rectangle. Defaults to None.
        north (int, optional): north coordinate to crop or locate rectangle. Defaults to None.
        crop_len (float, optional): crop or rectangle size. Defaults to None.
        as_rect (bool, optional): east, north, and crop_len to be used as rectangle coordinates. Defaults to False.
        lab (str, optional): either 'binary' or 'corrects'. Defaults to 'binary'.
    """

    BINARY_GROUPS, _, BINARY_COLORS = _get_binary_info(lab)
    
    BINARY_COLORS = ListedColormap(BINARY_COLORS)

    da = rioxarray.open_rasterio(fname, masked=True)
    dlabel, cmap_label = _get_ice_mask(fname, da)

    if east and north:
        if not as_rect:
            # use masks
            da = _crop_EE(da, east, north, crop_len)
            dlabel = _crop_EE(dlabel, east, north, crop_len)

    fig, ax = plt.subplots(figsize=(4.4,3.8))
    # Locating current axes
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
    
    da[0].plot.imshow(ax=ax, 
                    vmin=0,
                    vmax=len(BINARY_COLORS.colors),
                    interpolation='none', 
                    cmap=BINARY_COLORS, 
                    cbar_ax=cbar_ax)

    dlabel[0].plot.contour(ax=ax, 
                        cmap=cmap_label,
                        vmax=1,
                        linewidths=0.6, 
                        linestyles = 'dotted'
                        )
    
    cbar_ax.set_yticks(np.array(list(BINARY_GROUPS.keys()))+0.5)                              
    cbar_ax.set_yticklabels(list(BINARY_GROUPS.values()), rotation=90)
    cbar_ax.set_ylabel('')

    if as_rect:
        ax.add_patch(Rectangle((east-crop_len/2, north-crop_len/2), crop_len, crop_len,
             edgecolor = 'red',
             facecolor = 'blue',
             fill=False,
             lw=1))
        cbar_ax.cla()
        cbar_ax.axis('off')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('')
    fig.tight_layout()

    fig.savefig(os.path.join(dir_out, f'fig{fignumber}-{Path(fname).stem}.{file_format}'), dpi=dpi)
    plt.close('all')

def _predict_single(test_file, model_path):
    model = torch.jit.load(model_path)
    model.eval()

    label = 'binary'
    input_features = ['nersc_sar_primary',
                      'nersc_sar_secondary',
                      'sar_incidenceangle']

    print(f'Using raster {test_file}...', end=' ')
    
    scene = xr.open_dataset(test_file)

    if label == 'binary':
        scene['binary'] = scene['SIC'].copy()
        scene['binary'].values[scene['binary'].values > 1] = 1
        scene['binary'].values[scene['SIC'].values == scene['SIC'].chart_fill_value] = scene['SIC'].chart_fill_value

    x = torch.from_numpy(np.concatenate([np.expand_dims(scene[val], axis=0) for val in input_features], axis=0)).type(torch.float)
    x = x.unsqueeze(axis=0)
    y = scene[label].values

    softmax = nn.Softmax(0)
    with torch.no_grad():
        res = model(x)
        # compute probabilities (instead of scores):
        res = softmax(torch.squeeze(res,0)).detach().numpy()

    # compute entropy
    res_entropy = entropy(res, 
                    base=2, axis=0) 

    # write the class
    y_pred_class = res.argmax(0)

    corrects = y == y_pred_class

    # can't generate results with invalid input:
    mask_pred = scene[input_features[0]].values == scene[input_features[0]].variable_fill_value 
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

    scene[f'prob-{label}'] = scene[input_features[0]].copy()
    scene[f'prob-{label}'].values = res[1].copy()
    scene[f'prob-{label}'].values[~mask] = scene[input_features[0]].variable_fill_value

    scene['entropy'] = scene[input_features[0]].copy()
    scene['entropy'].values = res_entropy.copy()
    scene['entropy'].values[~mask] = scene[input_features[0]].variable_fill_value

    return scene

def fig1and2():

    fname = os.path.normpath('E:/rafael/data/AI4Arctic/version2/train/20200424T101936_cis_prep.nc')

    scene = xr.open_dataset(fname)

    ##################### plot input features
    _plot_input(scene, fname, 1)
    _plot_input_rgb(scene, fname, 1)

    ##################### plot label
    scene['binary'] = scene['SIC'].copy()
    scene['binary'].values[scene['binary'].values > 1] = 1
    scene['binary'].values[scene['SIC'].values == scene['SIC'].chart_fill_value] = scene['SIC'].chart_fill_value

    _plot_label(scene, fname, 2, 'binary')

def fig5():
    test_file = os.path.normpath('E:/rafael/data/AI4Arctic/version2/train/20201112T080407_dmi_prep.nc')
    model_path = os.path.normpath('E:/rafael/data/AI4Arctic/results/v1/dice/dice-5/model.pt')
    fignumber = 5

    scene = _predict_single(test_file, model_path)

    ##################### plot input features
    _plot_input(scene, test_file, fignumber)
    _plot_input_rgb(scene, test_file, f'{fignumber}-a')

    ##################### plot labels and corrects
    _plot_label(scene, test_file, fignumber, 'binary')
    _plot_label(scene, test_file, fignumber, 'pred-binary')
    _plot_label(scene, test_file, f'{fignumber}-d', 'corrects')

    ##################### plot ice probability and entropy
    _plot_mean_or_entropy(scene, test_file, f'{fignumber}-c', lab='entropy')
    _plot_mean_or_entropy(scene, test_file, f'{fignumber}-b', lab='prob-binary')

def fig6():
    test_file = os.path.normpath('E:/rafael/data/AI4Arctic/version2/train/20210715T211029_dmi_prep.nc')
    model_path = os.path.normpath('E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ce-7/model.pt')
    fignumber = 6
    scene = _predict_single(test_file, model_path)

    ##################### plot input features
    _plot_input(scene, test_file, fignumber)
    _plot_input_rgb(scene, test_file, fignumber)

    ##################### plot labels and corrects
    _plot_label(scene, test_file, f'{fignumber}-d', 'binary')
    _plot_label(scene, test_file, f'{fignumber}-e', 'pred-binary')
    _plot_label(scene, test_file, f'{fignumber}-f', 'corrects')

def fig11():
    fname = 'E:/rafael/data/Extreme_Earth/denoised_resampled/20180116t075430.tif'
    _plot_EE_rgb(fname, '11-a')

    fname = 'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ensemble_and_dropout_EE/pred-mean-20180116t075430.tif'
    _plot_EE_prob_or_entropy(fname, '11-b')

    fname = 'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ensemble_and_dropout_EE/pred-entropy-20180116t075430.tif'
    _plot_EE_prob_or_entropy(fname, '11-c')

def fig12():

    east, north = -502_600,-1_722_400
    crop_len = 150_000

    figtag = '20180911t175548'
    fignumber = 12

    fname = f'E:/rafael/data/Extreme_Earth/denoised_resampled/{figtag}.tif'
    _plot_EE_rgb(fname, f'{fignumber}-a', east, north, crop_len)

    fname = f'E:/rafael/data/Extreme_Earth/labels_rasterized/SA_wland/seaice_s1_{figtag}-SA_wland.tif'
    _plot_EE_icetype(fname, f'{fignumber}-b', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-dropout/pred-entropy-{figtag}.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-c', east, north, crop_len)

    # have to remake error map for figure 10:
    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ensemble_and_dropout_EE/corrects-{figtag}.tif'
    _plot_EE_label(fname, '10-09', east, north, crop_len, as_rect=True, lab='corrects')


def fig13():
    fignumber = 13
    fname = 'E:/rafael/data/Extreme_Earth/denoised_resampled/20180417t074606.tif'
    _plot_EE_rgb(fname, f'{fignumber}-a')

    fname = 'E:/rafael/data/Extreme_Earth/labels_rasterized/SA_wland/seaice_s1_20180417t074606-SA_wland.tif'
    _plot_EE_icetype(fname, f'{fignumber}-b')

    fname = 'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-dropout/pred-entropy-20180417t074606.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-c')

    fname = 'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-ensemble/pred-entropy-20180417t074606.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-d')

def fig14():
    east, north = -368297,-1646547
    crop_len = 80_000

    figtag = '20180515t174633'
    fignumber = 14

    fname = f'E:/rafael/data/Extreme_Earth/denoised_resampled/{figtag}.tif'
    _plot_EE_rgb(fname, f'{fignumber}-a', east, north, crop_len)

    fname = f'E:/rafael/data/Extreme_Earth/labels_rasterized/SA_wland/seaice_s1_{figtag}-SA_wland.tif'
    _plot_EE_icetype(fname, f'{fignumber}-b', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-dropout/pred-entropy-{figtag}.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-c', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-dropout/pred-mean-{figtag}.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-dropout/corrects-{figtag}.tif'
    _plot_EE_label(fname, f'{fignumber}-d', east, north, crop_len, lab='corrects')

    # have to remake error map for figure 10:
    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ensemble_and_dropout_EE/corrects-{figtag}.tif'
    _plot_EE_label(fname, '10-05', east, north, crop_len, as_rect=True, lab='corrects')


def fig15():
    east, north = -393_000,-1_479_000
    crop_len = 100_000

    figtag = '20180814t075344'
    fignumber = 15

    fname = f'E:/rafael/data/Extreme_Earth/denoised_resampled/{figtag}.tif'
    _plot_EE_rgb(fname, f'{fignumber}-a', east, north, crop_len)

    fname = f'E:/rafael/data/Extreme_Earth/labels_rasterized/SA_wland/seaice_s1_{figtag}-SA_wland.tif'
    _plot_EE_icetype(fname, f'{fignumber}-b', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ensemble_and_dropout_EE/corrects-{figtag}.tif'
    _plot_EE_label(fname, f'{fignumber}-c', east, north, crop_len, lab='corrects')

    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-dropout/pred-entropy-{figtag}.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-d', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-ensemble/pred-entropy-{figtag}.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-e', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ensemble_and_dropout_EE/pred-entropy-{figtag}.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-f', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/dice/EE-dropout/pred-entropy-{figtag}.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-g', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/dice/EE-ensemble/pred-entropy-{figtag}.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-h', east, north, crop_len)

    fname = f'E:/rafael/data/AI4Arctic/results/v1/dice/ensemble_and_dropout_EE/pred-entropy-{figtag}.tif'
    _plot_EE_prob_or_entropy(fname, f'{fignumber}-i', east, north, crop_len)

    # have to remake error map for figure 10:
    fname = f'E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ensemble_and_dropout_EE/corrects-{figtag}.tif'
    _plot_EE_label(fname, '10-08', east, north, crop_len, as_rect=True, lab='corrects')


if __name__ == '__main__':
    fig1and2()
    # # fig 3 is model architecture sketched on powerpoint
    # # fig 4 comes out of summaries.py (training_metrics.pdf)
    fig5() 
    fig6()
    # # fig 7 comes out of confusion matrix in evaluate.py 
    # # figs 8 and 9 come out of results in evaluate.py (ensemble dice and ensemble cross-entropy)
    # # fig 10 comes out of plot_raster.py (error maps)
    fig11()
    fig12()
    fig13()
    fig14()
    fig15()

