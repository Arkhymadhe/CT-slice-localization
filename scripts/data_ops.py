""" Data preparation operations. """

import os
import zlib
import zipfile

import pandas as pd
import numpy as np

import torch
from scipy.io import arff
from sklearn.model_selection import train_test_split


def read_compress(path_to_data, path_to_archive = 'compressed_data.zip'):
    """ Read data into memory; compress and delete raw file."""

    df = read_dataset(path_to_data)

    with zipfile.ZipFile(path_to_archive, 'w') as zf:
        zf.write(path_to_data, arcname = 'ct_dataset.csv', compress_type = zipfile.ZIP_DEFLATED)
        zf.close()

    os.remove(path_to_data)

    return df, path_to_archive


def read_dataset(data_dir : str) -> pd.DataFrame:
    ''' Load dataset into memory. '''
    
    data = arff.loadarff(data_dir)
    data = pd.DataFrame(data[0])
    
    return data


def variables(data, labels, target : str, axis = 1, return_y = True):
    ''' Return covariates and target. '''
    
    assert type(target) in [str, list], 'Target must be a feature name  or list of feature names.'
    
    duplicates = data.duplicated()
    print(f'Number of duplicate records: {duplicates.shape[0]}')
    
    data = data.drop_duplicates()
    data = data.dropna(axis = 0, subset = target)
    
    X = data.drop(labels = labels, axis = axis)
    
    print(f'Number of observations: {X.shape[0]}',
          f'Feature dimensionality: {X.shape[1]}')
    
    if return_y:
        y = data.loc[:, target]
        
        if type(target) is str:
            y = y.values.reshape(X.shape[0], 1)
        else:
            y = y.values.reshape(X.shape[0], len(target))
        
        print(f'Number of available targets: {y.shape[0]}',
              f'Target variable(s): {y.shape[1]}')
    else:
        y = None
        print('No target variables returned')
    
    return X, y if y else X


def get_invariant_features(X : pd.DataFrame, cardinality = 50, percent : float = 0.8):
    """ Obtain features below cutoff variance threshold. """
    
    ### Container for invariant features
    to_drop = []
    
    ### Possible catregorical features
    cat_features = X.nunique()[X.nunique() <= cardinality].index

    for feature in cat_features:
        counts = X[feature].value_counts(normalize = True)
        print(f'Feature diagnostics (Feature `{feature})` :')

        if counts.max() > percent:
            to_drop.append(feature)

        for c in counts.index:
            print(f'\t{c} has a count(s) of {counts[c] : .3f}')

        print()
        print('+'*(100))
        print()
        
    print(f'Total number of features with substandard variance: {len(to_drop)}')
    
    return to_drop


def array_to_tensor(array):
    """ Convert NumPy array to torch.Tensor. """
    assert type(array) in [pd.DataFrame, pd.Series, np.ndarray], 'Array must be of type `pd.DataFrame`, `pd.Series`, or `np.ndarray`.'
    return torch.from_numpy(array).to(torch.float32)


def split_data(X, y, split_size=0.2):
    """ Split dataset for model evaluation and testing. """

    X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=split_size, stratify=y)

    return X_1, X_2, y_1, y_2