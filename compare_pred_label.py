"""
Compare ice-water prediction with ice type and concentration
"""
import os
from configparser import ConfigParser
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
import torch
import xarray as xr
from torch import nn
from tqdm.auto import tqdm


#https://gist.github.com/Mlawrence95/f697aa939592fa3ef465c05821e1deed
def confusion_matrix(df: pd.DataFrame, col1: str, col2: str):
    """
    Given a dataframe with at least
    two categorical columns, create a 
    confusion matrix of the count of the columns
    cross-counts
    
    use like:
    
    >>> confusion_matrix(test_df, 'actual_label', 'predicted_label')
    """
    return (
            df
            .groupby([col1, col2])
            .size()
            .unstack(fill_value=0)
            )

def compare_EE():
    dirs_in = ['E:/rafael/data/AI4Arctic/results/v1/cross-entropy/ensemble_and_dropout_EE']
    
    dir_in_icetype = os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/SA_wland')

    dir_in_concentration = os.path.normpath('E:/rafael/data/Extreme_Earth/labels_rasterized/CT')
    
    dfs = []
    
    # land values
    land_icetype = 6
    # right now concentration has land and water with the same values
    land_ct = 23

    # value dictionaries
    ice_water_dict = {0:'W', 
                      1:'Ice'}
    type_dict = {0:'NI', 
                 1:'Nilas', 
                 2:'YI' ,
                 3:'FYI', 
                 4:'OI',
                 5:'W' }

    concentration_dict = {0:'less than 1 tenths', 
                          1:1, 
                          2:2, 
                          3:'1 to 2 tenths', 
                          4:'1 to 3 tenths', 
                          5:'2 to 3 tenths', 
                          6:'2 to 4 tenths', 
                          7:'3 to 4 tenths', 
                          8:'3 to 5 tenths', 
                          9:'3 tenths', 
                          10:'4 to 5 tenths', 
                          11:'4 to 6 tenths', 
                          12:'4 tenths', 
                          13:'5 to 7 tenths', 
                          14:'5 tenths', 
                          15:'6 to 8 tenths', 
                          16:'7 to 8 tenths', 
                          17:'7 to 9 tenths', 
                          18:'8 to 10 tenths', 
                          19:'9 tenths', 
                          20:'8 to 9 tenths', 
                          21:'9+ tenths', 
                          22:'10 tenths', 
                          23:'W'
                          }

    for dir_in in dirs_in:
        dir_in = os.path.normpath(dir_in)

        pred_class_files = [f for f in os.listdir(dir_in) if f.endswith('.tif') and f.startswith('class')]
        
        for pred_class_file in tqdm(pred_class_files):
            scene_id = Path(pred_class_file).stem.split('-')[1]
            icetype_filename = f'seaice_s1_{scene_id}-SA_wland.tif'
            concentration_filename = f'seaice_s1_{scene_id}-CT.tif'
            
            preds = rioxarray.open_rasterio(os.path.join(dir_in, pred_class_file), masked=True).data.ravel()
            icetype = rioxarray.open_rasterio(os.path.join(dir_in_icetype, icetype_filename), masked=True).data.ravel()
            ct = rioxarray.open_rasterio(os.path.join(dir_in_concentration, concentration_filename), masked=True).data.ravel()

            df = pd.DataFrame(list(zip(preds, icetype, ct)), columns=['pred', 'SA', 'CT'])
            df.loc[df['SA']==land_icetype, 'SA'] = np.nan

            df = df.dropna(axis='rows')
            df['scene'] = scene_id
            
            dfs.append(df)
        

        df = pd.concat(dfs)
        # use names instead of values:
        df = df.replace({'pred': ice_water_dict, 'SA':type_dict, 'CT':concentration_dict})

        cm = confusion_matrix(df, 'pred', 'SA')
        cm.to_csv(os.path.join(dir_in, 'full_confusion_matrix-type.csv'))

        cm = confusion_matrix(df, 'pred', 'CT')
        cm.to_csv(os.path.join(dir_in, 'full_confusion_matrix-ct.csv'))

def compare_ai4arctic():

    dirs_in = ['E:/rafael/data/AI4Arctic/results/v1/cross-entropy/primary_ensemble']
    
    dfs = []
    
    # value dictionaries
    ice_water_dict = {0:'W', 
                      1:'Ice'}

    SIC_GROUPS = {
        0: 0,
        1: 10,
        2: 20,
        3: 30,
        4: 40,
        5: 50,
        6: 60,
        7: 70,
        8: 80,
        9: 90,
        10: 100
    }

    SOD_GROUPS = {
        0: 'Open water',
        1: 'New Ice',
        2: 'Young ice',
        3: 'Thin FYI',
        4: 'Thick FYI',
        5: 'Old ice',
    }

    for dir_in in dirs_in:
        dir_in = os.path.normpath(dir_in)

        config = ConfigParser()
        config.read(os.path.join(dir_in, 'evaluate-test.cfg'))

        dir_nc = os.path.normpath(config['io']['dir_in'])
        
        if 'file_list' in config['io']:
            test_files = [os.path.join(dir_nc, f) for f in config['io']['file_list'].split('\n')]
        else:
            test_files = [os.path.join(dir_nc, f) for f in os.listdir(dir_nc) if f.endswith('.nc')]

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

        input_features = ['nersc_sar_primary',
                        'nersc_sar_secondary',
                        'sar_incidenceangle']

        # run on test rasters:
        softmax = nn.Softmax(0)
        for test_file in tqdm(test_files):

            scene = xr.open_dataset(test_file)

            x = torch.from_numpy(np.concatenate([np.expand_dims(scene[val], axis=0) for val in input_features], axis=0)).type(torch.float)
            x = x.unsqueeze(axis=0)
            
            if ensemble:
                res = []
                for model in models:
                    res.append(softmax(torch.squeeze(model(x).detach(),0)))
            
                res_mean = torch.mean(torch.stack(res), dim=0).detach().numpy()
                res = res_mean

            else:
                with torch.no_grad():
                    res = model(x)
                    # compute probabilities (instead of scores):
                    res = softmax(torch.squeeze(res,0)).detach().numpy()
            
            preds = res.argmax(0).ravel().astype(float)
            sod = scene['SOD'].data.ravel().astype(float)
            sic = scene['SIC'].data.ravel().astype(float)

            df = pd.DataFrame(list(zip(preds, sod, sic)), columns=['pred', 'SOD', 'SIC'])
            df.loc[df['SOD']==255, 'SOD'] = np.nan
            df.loc[df['SIC']==255, 'SIC'] = np.nan

            df = df.dropna(axis='rows')
            df['scene'] = Path(test_file).stem
            
            dfs.append(df)
        
        df = pd.concat(dfs)
        # use names instead of values:
        df = df.replace({'pred': ice_water_dict, 'SOD':SOD_GROUPS, 'SIC':SIC_GROUPS})

        cm = confusion_matrix(df, 'pred', 'SOD')
        cm.to_csv(os.path.join(dir_in, 'full_confusion_matrix-sod.csv'))

        cm = confusion_matrix(df, 'pred', 'SIC')
        cm.to_csv(os.path.join(dir_in, 'full_confusion_matrix-sic.csv'))


if __name__ == '__main__':
    #compare_EE()
    compare_ai4arctic()