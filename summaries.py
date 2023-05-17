import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def metric_summary():

    root_dir = os.path.normpath("E:/rafael/data/AI4Arctic/results/v1")

    list_test_df = []
    list_validation_df = []

    for fname in Path(root_dir).rglob('*metrics*'):
        if ('test-summary_metrics.csv' in os.path.normpath(fname).split(os.sep)):
            list_test_df.append(pd.read_csv(fname, index_col=0))
            list_test_df[-1]['experiment'] = os.path.normpath(fname).split(os.sep)[-2]
            list_test_df[-1]['group'] = list_test_df[-1]['experiment'].str.split('-', expand=True)[0]

        if ('validation-summary_metrics.csv' in os.path.normpath(fname).split(os.sep)):
            list_validation_df.append(pd.read_csv(fname, index_col=0))
            list_validation_df[-1]['experiment'] = os.path.normpath(fname).split(os.sep)[-2]
            list_validation_df[-1]['group'] = list_validation_df[-1]['experiment'].str.split('-', expand=True)[0]

    test_df = pd.concat(list_test_df)
    test_df.index.name='scene'
    validation_df = pd.concat(list_validation_df)
    validation_df.index.name='scene'

    test_df.to_csv(os.path.join(root_dir, 'test-compilation.csv'))
    validation_df.to_csv(os.path.join(root_dir, 'validation-compilation.csv'))

    # group and save aggregate statistics
    agg_func_math = {'accuracy': ['median', 'min', 'max', 'mean','std'],
                     'iou-micro': ['median', 'min', 'max', 'mean','std'],
                     'iou-macro': ['median', 'min', 'max', 'mean','std'],
                     'iou-weighted': ['median', 'min', 'max', 'mean','std'],
                     'f1-micro': ['median', 'min', 'max', 'mean','std'],
                     'f1-macro': ['median', 'min', 'max', 'mean','std'],
                     'f1-weighted': ['median', 'min', 'max', 'mean','std'],

                     }

    test_df.groupby(['group', 'scene']).agg(agg_func_math).sort_values(['group', 'scene']).to_csv(os.path.join(root_dir, 'summary_grouped_by_scene_test.csv'))
    test_df.groupby(['experiment']).agg(agg_func_math).sort_values(['experiment']).to_csv(os.path.join(root_dir, 'summary_grouped_by_experiment_test.csv'))
    validation_df.groupby(['experiment']).agg(agg_func_math).sort_values(['experiment']).to_csv(os.path.join(root_dir, 'summary_grouped_validation.csv'))

def full_loss_decay():

    root_dirs = ["E:/rafael/data/AI4Arctic/results/v1"
                ]

    for root_dir in root_dirs:
    
        df_list = []

        for fname in Path(root_dir).rglob('*metrics*'):

            if set(['lightning_logs', 'version_1']).issubset(set(os.path.normpath(fname).split(os.sep))):
                print(fname)

                df_list.append(pd.read_csv(fname))
                df_list[-1]['experiment_full'] = fname
                df_list[-1]['experiment_number'] = os.path.normpath(fname).split(os.sep)[-4]
                df_list[-1]['Loss'] = os.path.normpath(fname).split(os.sep)[-5]

        df = pd.concat(df_list).reset_index(drop=True)
        
        # create dset split column and remove nan vals
        df['Set'] = np.nan
        for dset in ['val_loss', 'train_loss_epoch']:
            df.loc[~df[dset].isnull(), 'Set'] = dset.split('_')[0]
        
        df = df[~df['Set'].isnull()].dropna(axis=1, how='all').drop('step', axis=1)

        cols = [i for i in df.columns if i not in ['Set', 'experiment', 'epoch', 'experiment_full', 'experiment_number', 'Loss']]

        df_long = df.melt(id_vars=['epoch', 'Set', 'experiment_full', 'experiment_number', 'Loss'], value_vars=cols).dropna(axis=0, how='any')

        # generate loss plot
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=(4.5,4.5), ncols=1) 

            metric = 'Loss'
                
            df_long_select = df_long[df_long['variable'].str.lower().str.contains(metric.split(' ')[0].lower())].copy()
            df_long_select['variable'] = df_long_select['variable'].str.split('_', expand=True)[0]

            g = sns.lineplot(x="epoch", y="value",
                        style="Set",
                        hue='Loss',
                        ax = ax,
                        estimator='mean', #np.median,
                        ci=99, #lambda x: (np.min(x), np.max(x)),
                        data=df_long_select)
                
            ax.set(yscale="log")                        
            ax.set_ylim(bottom=0.01)
            ax.set_ylabel('Loss value')
            ax.set_xlabel('Epoch')
        
        fig.tight_layout()
        fig.savefig(os.path.join(root_dir, 'training_metrics.pdf'), dpi=500)


if __name__ == '__main__':
    
    full_loss_decay()
    #metric_summary()