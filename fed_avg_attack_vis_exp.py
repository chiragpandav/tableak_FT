import torch
import utils
from models import FullyConnected
import attacks
from utils import get_acc_and_bac
from datasets import ADULT, Lawschool
import numpy as np
import copy
from attacks.fed_avg_inversion_attack import train_and_attack_fed_avg
from utils import match_reconstruction_ground_truth
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd

experiments = {
    0: 'Inverting Gradients',
    52: 'TabLeak'
}

dataset_names = ['ADULT', 'German', 'HealthHeritage', 'Lawschool']  # set to include only the datasets on which you have already obtained the data

possible_num_epochs = np.array([1, 5, 10])
possible_num_batches = np.array([1, 2, 4])
possible_params = np.array([0.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001])

dataframes_datasets = {}

for dataset_name in dataset_names:
    
    dataframes_experiments = {}
    
    for k, (experiment_num, experiment_name) in enumerate(experiments.items()):
        path = f'experiment_data/fedavg_experiments/{dataset_name}/experiment_{experiment_num}/inversion_data_all_{experiment_num}_{dataset_name}_50_{list(possible_params)}_0.319_42.npy'
        print(path)
        if osp.isfile(path):
            data = np.load(path)
        else:
            continue
        
        one_epoch = []
        five_epochs = []
        ten_epochs = []
        for j, n_batches in enumerate(possible_num_batches):
            index_of_best_param = np.argmin(data[:, j, :, 0, 0], axis=-1)
            best_mean = 100 - 100*data[np.arange(len(index_of_best_param)), j, index_of_best_param, 0, 0]
            best_std = 100*data[np.arange(len(index_of_best_param)), j, index_of_best_param, 0, 1]
            best_param = possible_params[index_of_best_param]
            data_summary = [(np.around(mean, 2), np.around(std, 2), param) for mean, std, param in zip(best_mean, best_std, best_param)]
            one_epoch.append((np.around(data_summary[0][0], 1), np.around(data_summary[0][1], 1)))
            five_epochs.append((np.around(data_summary[1][0], 1), np.around(data_summary[1][1], 1)))
            ten_epochs.append((np.around(data_summary[2][0], 1), np.around(data_summary[2][1], 1)))
        
        df = pd.DataFrame({'N. Batches': [1, 2, 4], '1 Epoch': one_epoch, '5 Epochs': five_epochs, '10 Epochs': ten_epochs})
        dataframes_experiments[experiment_name] = df
        print(df)
    dataframes_datasets[dataset_name] = dataframes_experiments

# print(dataframes_datasets['ADULT']['TabLeak'])
print(dataframes_datasets)