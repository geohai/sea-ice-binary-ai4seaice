import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import rioxarray
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler


def plot_input():
    
    root_dir = os.path.normpath('E:/rafael/data/Extreme_Earth/denoised_resampled')
    for fname in Path(root_dir).rglob('*.tif*'):
        print(fname)
        da = rioxarray.open_rasterio(fname, masked=True)

        # clip 2-98% and rescale
        scaler = MinMaxScaler()
        for idx, band in enumerate(da.data):
            clip_min, clip_max = np.nanpercentile(band, [2,98])
            da.data[idx] = scaler.fit_transform(np.clip(band, clip_min, clip_max).reshape(-1,1)).reshape(band.shape)

        fig, ax = plt.subplots(figsize=(5,5))
        da.plot.imshow(ax=ax, rgb='band')
        ax.set_title('')
        ax.axis('off')
        fig.tight_layout()

        fig.savefig(os.path.join(root_dir, f'{Path(fname).stem}.png'), dpi=400)

        plt.close('all')

def plot_results_or_label():

    root_dirs = [os.path.normpath('E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ensemble_and_dropout_EE'), 
                 #os.path.normpath('E:/rafael/data/AI4Arctic/results/v1/dice/ensemble_and_dropout_EE'), 
                 os.path.normpath('E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-dropout'),
                 os.path.normpath('E:/rafael/data/AI4Arctic/results/v1/cross-entropy/EE-ensemble')
                 ]
    root_label = os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/poly_type_wland')
    label_tag = 'poly_type'
    num_classes = 1

    for root_dir in root_dirs:
        for fname in Path(root_dir).rglob('*.tif*'):
            
            # this file should be associated with a label:
            label_fname = os.path.join(root_label, f'seaice_s1_{Path(fname).stem.split("-")[-1]}-{label_tag}.tif')
            dlabel = rioxarray.open_rasterio(label_fname, masked=True)
            dlabel.data[dlabel.data > num_classes] = num_classes
            cmap_label = ListedColormap([(255/255, 255/255, 255/255)])

            fig, ax = plt.subplots(figsize=(4,4))

            if Path(fname).stem.startswith('class-') or Path(fname).stem.startswith('corrects-'):
                print(fname)
                da = rioxarray.open_rasterio(fname, masked=True)

                if 'class' in os.path.normpath(fname):
                    cmap = ListedColormap(['#67a9cf', '#ef8a62', (196/255, 196/255, 196/255)])
                
                if Path(fname).stem.startswith('corrects-'):
                    cmap = ListedColormap(['#7570b3', '#1b9e77', (0/255, 255/255, 0/255)])

                da[0].plot(ax=ax, 
                            vmin=0,
                            vmax=len(cmap.colors)-1,
                            add_colorbar=False, 
                            cmap=cmap)

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
                fig.savefig(os.path.join(root_dir, f'{Path(fname).stem}.png'), dpi=400)

                plt.close('all')

            if Path(fname).stem.startswith('pred-'):
                print(fname)
                da = rioxarray.open_rasterio(fname, masked=True)

                if 'mean' in os.path.normpath(fname):
                    # get the mask that includes land:
                    mask = np.isnan(rioxarray.open_rasterio(os.path.join(root_dir,
                                                                         f"class-{Path(fname).stem.split('-')[-1]}.tif"), 
                                                                         masked=True))
                    # set the mask:
                    da[1] = da[1].where(~mask[0], np.nan)

                    # Locating current axes
                    divider = make_axes_locatable(ax)
                    cbar_ax = divider.append_axes("right",
                                                    size="10%",
                                                    pad=0.1)

                    da[1].plot(ax=ax, cbar_ax=cbar_ax, vmin=0.2, vmax=1.0, cmap='viridis')

                    
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
                    fig.savefig(os.path.join(root_dir, f'{Path(fname).stem}.png'), dpi=400)
                
                if 'var' in os.path.normpath(fname):
                    # get the mask that includes land:
                    mask = np.isnan(rioxarray.open_rasterio(os.path.join(root_dir,
                                                                         f"class-{Path(fname).stem.split('-')[-1]}.tif"), 
                                                                         masked=True))
                    # set the mask:
                    da[1] = da[1].where(~mask[0], np.nan)

                    # Locating current axes
                    divider = make_axes_locatable(ax)
                    cbar_ax = divider.append_axes("right",
                                                    size="10%",
                                                    pad=0.1)

                    da[1].plot(ax=ax, cbar_ax=cbar_ax, cmap='viridis', robust = True)
                    
                    dlabel[0].plot.contour(ax=ax, 
                                            cmap=cmap_label,
                                            vmax=1,
                                            linewidths=0.6, 
                                            linestyles = 'dotted'
                                            )

                    ax.set_title('')
                    ax.set_aspect('equal')
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                    fig.tight_layout()
                    fig.savefig(os.path.join(root_dir, f'{Path(fname).stem}.png'), dpi=400)

                if 'entropy' in Path(os.path.normpath(fname)).stem:
                    # get the mask that includes land:
                    mask = np.isnan(rioxarray.open_rasterio(os.path.join(root_dir,
                                                                         f"class-{Path(fname).stem.split('-')[-1]}.tif"), 
                                                                         masked=True))
                    # set the mask:
                    da[0] = da[0].where(~mask[0], np.nan)

                    # Locating current axes
                    divider = make_axes_locatable(ax)
                    cbar_ax = divider.append_axes("right",
                                                    size="10%",
                                                    pad=0.1)

                    da[0].plot(ax=ax, cbar_ax=cbar_ax, cmap='inferno', vmin=0.29, vmax=1.0)
                    
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
                    fig.savefig(os.path.join(root_dir, f'{Path(fname).stem}.png'), dpi=400)


                plt.close('all')


def plot_land():

    root_dirs = [os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/poly_type_wland'),
                 ]

    for root_dir in root_dirs:
        for fname in Path(root_dir).rglob('*.tif*'):

            if Path(fname).stem.startswith('class-') or Path(fname).stem.startswith('correct-') or Path(fname).stem.startswith('seaice'):
                print(fname)
                da = rioxarray.open_rasterio(fname, masked=True)

                if 'poly_type' in os.path.normpath(fname):
                    cmap = ListedColormap([(51/255, 153/255, 255/255, 0), (255/255, 125/255, 7/255, 0), (196/255, 196/255, 196/255)])
                
                elif ('SA' in os.path.normpath(fname)) or ('SD' in os.path.normpath(fname)):
                    cmap = ListedColormap([(253/255, 204/255, 224/255), 
                                        (152/255, 111/255, 196/255), 
                                        (228/255, 0/255, 217/255), 
                                        (250/255, 243/255, 13/255), 
                                        (231/255, 61/255, 4/255), 
                                        (51/255, 153/255, 255/255), 
                                        (196/255, 196/255, 196/255)])           

                if Path(fname).stem.startswith('correct-'):
                    cmap = ListedColormap([(220/255, 16/255, 203/255), (0/255, 255/255, 0/255), (0/255, 255/255, 0/255)])

                fig, ax = plt.subplots(figsize=(5,5))
                da[0].plot.imshow(ax=ax, 
                                  vmin=0,
                                  vmax=len(cmap.colors)-1,
                                  add_colorbar=False, 
                                  interpolation='none', 
                                  cmap=cmap)
                ax.set_title('')
                ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(root_dir, f'land-{Path(fname).stem}.png'), dpi=400)

                plt.close('all')

def plot_pixel_count_per_class():

    root_dirs = [os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/SA_wland'),
                 os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/SD_wland'),
                 ]
    root_names = ['Oldest Ice Type', 'Dominant Ice Type']
    dir_out = 'C:/Users/rafael/Desktop/paper_figures'

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(7, 7), ncols=2) 
    
        for idx, (root_dir, root_name, letter) in enumerate(zip(root_dirs, root_names, ['a)', 'b)'])):
            dfs = []
            for fname in Path(root_dir).rglob('*.tif*'):
                if Path(fname).stem.startswith('class-') or Path(fname).stem.startswith('correct-') or Path(fname).stem.startswith('seaice'):
                    if 'NE' not in Path(fname).stem and 'SW' not in Path(fname).stem:
                        print(fname) 
                        da = rioxarray.open_rasterio(fname, masked=True)

                        unique, counts = np.unique(da.data.ravel(), return_counts=True)
                        month = datetime.strptime(Path(fname).stem.split('-')[0].split('_')[-1].split('t')[0], '%Y%m%d').strftime('%B')
                        dfs.append(pd.DataFrame(data={month:counts}, index=unique))

            df = pd.concat(dfs, axis=1)
            df = df.rename(index={0.0: 'New Ice', 
                                1.0: 'Nilas',
                                2.0: 'Young Ice', 
                                3.0: 'First Year Ice',
                                4.0: 'Old Ice', 
                                5.0: 'Water', 
                                6.0: 'Land'})
            df = df.drop(np.NaN, axis=0)

            df.to_csv(os.path.join(dir_out, f'{root_name}.csv'))
            
            df.transpose().plot(kind='bar', stacked=True, 
                                        color=[(253/255, 204/255, 224/255), 
                                        (152/255, 111/255, 196/255), 
                                        (228/255, 0/255, 217/255), 
                                        (250/255, 243/255, 13/255), 
                                        (231/255, 61/255, 4/255), 
                                        (51/255, 153/255, 255/255), 
                                        (196/255, 196/255, 196/255)], 
                                        ax=ax[idx])
            ax[idx].set_title(root_name)
            ax[idx].xaxis.grid(False)
            ax[idx].text(-0.15, 1.05, letter, size=14, transform=ax[idx].transAxes)
            ax[idx].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1e6) + 'M'))

        ax[0].get_legend().remove()
        fig.tight_layout()
        fig.savefig(os.path.join(dir_out, 'pixel_count.pdf'))
       

if __name__ == '__main__':
    
    #plot_input()
    plot_results_or_label()
    #plot_land()
    #plot_pixel_count_per_class()
