#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import attacks
import numpy as np
import torch
# from utils.matching_modified_1 import match_reconstruction_ground_truth
from utils.matching import match_reconstruction_ground_truth
import pandas as pd
from utils import Timer, post_process_continuous
from attacks import train_and_attack_fed_avg
from models import FullyConnected

import argparse
import pickle
from attacks import calculate_random_baseline
# from datasets.adult_modified import ADULT
from datasets.base_dataset import BaseDataset


# In[2]:


configs = {
    # Inverting Gradients
    0: {
        'n_global_epochs': 1,
        'lr': 0.01,
        'shuffle': True,
        'attacked_clients': 'all',
        'attack_iterations': 1500,
        'reconstruction_loss': 'cosine_sim',
        'priors': None,
        'epoch_matching_prior': 'mean_squared_error',
        'post_selection': 1,
        'attack_learning_rate': 0.06,
        'return_all': False,
        'pooling': None,
        'perfect_pooling': False,
        'initialization_mode': 'uniform',
        'softmax_trick': False,
        'gumbel_softmax_trick': False,
        'sigmoid_trick': False,
        'temperature_mode': 'constant',
        'sign_trick': True,
        'verbose': False,
        'max_client_dataset_size': 32,
        'post_process_cont': False
    },
    # TabLeak
    52: {
        'n_global_epochs': 1,
        'lr': 0.01,
        'shuffle': True,
        'attacked_clients': 'all',
        'attack_iterations': 1500,
        'reconstruction_loss': 'cosine_sim',
        'priors': None,
        'epoch_matching_prior': 'mean_squared_error',
        'post_selection': 15,
        'attack_learning_rate': 0.06,
        'return_all': False,
        'pooling': 'median',
        'perfect_pooling': False,
        'initialization_mode': 'uniform',
        'softmax_trick': True,
        'gumbel_softmax_trick': False,
        'sigmoid_trick': True,
        'temperature_mode': 'constant',
        'sign_trick': True,
        'verbose': False,
        'max_client_dataset_size': 32,
        'post_process_cont': False
    }
}


# In[3]:


# client_models = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
#                "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
#                "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
#                "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
#                "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

client_models = ["AL"]

# testing_data=["WM","WW","BM","BW"]


# In[4]:


gt_info=pd.read_csv('50_clients_data/state_sex_race_counts.csv')
gt_info.head()


# In[5]:


def group_filter(client_gt_projected, client_recon_projected, index, value):
    # print("group_filter applied")
    print("index: ",index, "value: ",value)
    filtered_gt = client_gt_projected[client_gt_projected[:, index] == value]
    filtered_recon = client_recon_projected[client_gt_projected[:, index] == value]
    return filtered_gt, filtered_recon


def subgroup_filter(client_gt_projected, client_recon_projected, index1, value1, index2, value2):
    # print("subgroup_filter applied")
    print("index1: ", index1,"value1: ", value1, "index2: ",index2,"value2: ", value2)
    condition = (client_gt_projected[:, index1] == value1) & (client_gt_projected[:, index2] == value2)
    filtered_gt = client_gt_projected[condition]
    filtered_recon = client_recon_projected[condition]
    return filtered_gt, filtered_recon


# In[ ]:


exp1 = "tableak_FT_inversion_normal"
config = configs[0]
final_grp={}
final_subgrp={}

print("Inversion Normal attack")
num_training_datapoints=2000

final_all_errors = []
final_cat_errors = []
final_cont_errors = []

final_all_errors_full = []
final_cat_errors_full = []
final_cont_errors_full = []


for model_name in client_models:
    print("----------- Model_name --------------: ",model_name)
    temp_result={}
    
    recn_gt = f'../{exp1}/50_clients_data/reconstr_and_GT/reconstructions_ground_truths_{model_name}.pkl'
    dataset = f'../{exp1}/50_clients_data/reconstr_and_GT/dataset_{model_name}.pkl'
    tolerance_map = f'../{exp1}/50_clients_data/reconstr_and_GT/tolerance_map_{model_name}.pkl'

    with open(recn_gt, 'rb') as file:
        recn_gt = pickle.load(file)

    with open(dataset, 'rb') as file:
        dataset = pickle.load(file)
    
    with open(tolerance_map, 'rb') as file:
        tolerance_map = pickle.load(file)

    reconstructions = recn_gt['reconstructions']
    ground_truths = recn_gt['ground_truths']

    all_errors = []
    cat_errors = []
    cont_errors = []
    
    all_errors_full = []
    cat_errors_full = []
    cont_errors_full = []
    
    temp_all_errors = []
    temp_cat_errors = []
    temp_cont_errors = []
    
    match_results = {}
    index_value_pairs = [(8, 1),(8, 2), (9, 1), (9, 2)]

    for epoch_reconstruction, epoch_ground_truth in zip(reconstructions, ground_truths):
        for client_reconstruction, client_ground_truth in zip(epoch_reconstruction, epoch_ground_truth):
            if config['post_process_cont']:
                client_reconstruction = post_process_continuous(client_reconstruction, dataset=dataset)
            client_recon_projected, client_gt_projected = dataset.decode_batch(client_reconstruction, standardized=True), dataset.decode_batch(client_ground_truth, standardized=True)
                    
            print(client_gt_projected.shape, client_recon_projected.shape)
                    
            _, batch_cost_all_original, batch_cost_cat_original, batch_cost_cont_original = match_reconstruction_ground_truth(
                                client_gt_projected, client_recon_projected, tolerance_map
                                )
            all_errors_full.append(np.mean(batch_cost_all_original))
            cat_errors_full.append(np.mean(batch_cost_cat_original))
            cont_errors_full.append(np.mean(batch_cost_cont_original))  

            final_all_errors_full.append(all_errors_full)
            final_cat_errors_full.append(cat_errors_full)
            final_cont_errors_full.append(cont_errors_full) 
            
            for index, value in index_value_pairs:

                temp_all_errors = []
                temp_cat_errors = []
                temp_cont_errors = []
            
                filtered_gt, filtered_recon = group_filter(client_gt_projected, client_recon_projected, index=index, value=value)        
                print(filtered_gt.shape, filtered_recon.shape)
    
                # Call match_reconstruction_ground_truth with the filtered results
                _, batch_cost_all, batch_cost_cat, batch_cost_cont = match_reconstruction_ground_truth(filtered_gt, filtered_recon, tolerance_map)
                
                temp_all_errors.append(np.mean(batch_cost_all))
                temp_cat_errors.append(np.mean(batch_cost_cat))
                temp_cont_errors.append(np.mean(batch_cost_cont))  
    
                all_errors.append(temp_all_errors)
                cat_errors.append(temp_cat_errors)
                cont_errors.append(temp_cont_errors) 
                
            final_all_errors.append(all_errors)
            final_cat_errors.append(cat_errors)
            final_cont_errors.append(cont_errors)         


# In[ ]:


final_all_errors_full


# In[ ]:





# In[ ]:





# In[ ]:


local_epochs=[5]
local_batch_sizes=[8]
epoch_prior_params=[0.01]
all_state_recon={}

for m,model_name in enumerate(client_models):
    collected_data = np.zeros((len(local_epochs), len(local_batch_sizes), len(epoch_prior_params), 3, 5))
    
    for i, lepochs in enumerate(local_epochs):
        for j, lbatch_size in enumerate(local_batch_sizes):
            for k, epoch_prior_param in enumerate(epoch_prior_params):
    
                collected_data[i, j, k, 0] = np.mean(final_all_errors_full[m]), np.std(final_all_errors_full[m]), np.median(final_all_errors_full[m]), np.min(final_all_errors_full[m]), np.max(final_all_errors_full[m])
                collected_data[i, j, k, 1] = np.mean(final_cat_errors_full[m]), np.std(final_cat_errors_full[m]), np.median(final_cat_errors_full[m]), np.min(final_cat_errors_full[m]), np.max(final_cat_errors_full[m])
                collected_data[i, j, k, 2] = np.mean(final_cont_errors_full[m]), np.std(final_cont_errors_full[m]), np.median(final_cont_errors_full[m]), np.min(final_cont_errors_full[m]), np.max(final_cont_errors_full[m])
    
            best_param_index = np.argmin(collected_data[i, j, :, 0, 0]).item()

    temp_var=float(100 * (1 - collected_data[i, j, best_param_index, 0, 0]))
    all_state_recon[model_name]=temp_var
    # print(f'Performance at {lepochs} Epochs and {lbatch_size} Batch Size: {100*(1-collected_data[i, j, best_param_index, 0, 0]):.1f}% +- {100*collected_data[i, j, best_param_index, 0, 1]:.2f}')
    # total_recon= 100 * (1 - collected_data[i, j, best_param_index, 0, 0])
    # print("total_recon: ",total_recon)


# In[ ]:


all_state_recon


# In[ ]:


final_all_errors


# In[ ]:





# In[ ]:





# In[ ]:


local_epochs = [5]
local_batch_sizes = [8]
epoch_prior_params = [0.01]
final_group_reco={}
for m,model_name in enumerate(client_models):
    
    temp_group_score=[]
    all_errors =final_all_errors[m]
    cat_errors=final_cat_errors[m]
    cont_errors=final_cont_errors[m]
    print(all_errors)
    for i in range(len(all_errors)):
        collected_data = np.zeros((len(local_epochs), len(local_batch_sizes), len(epoch_prior_params), 3, 5))

        for m, lepochs in enumerate(local_epochs): 
            for j, lbatch_size in enumerate(local_batch_sizes):
                for k, epoch_prior_param in enumerate(epoch_prior_params):
                    
                    collected_data[m, j, k, 0] = np.mean(all_errors[i]), np.std(all_errors[i]), np.median(all_errors[i]), np.min(all_errors[i]), np.max(all_errors[i])
                    collected_data[m, j, k, 1] = np.mean(cat_errors[i]), np.std(cat_errors[i]), np.median(cat_errors[i]), np.min(cat_errors[i]), np.max(cat_errors[i])
                    collected_data[m, j, k, 2] = np.mean(cont_errors[i]), np.std(cont_errors[i]), np.median(cont_errors[i]), np.min(cont_errors[i]), np.max(cont_errors[i])
    
                best_param_index = np.argmin(collected_data[m, j, :, 0, 0]).item()
    
            # print("best_param_index", best_param_index)
            # print(f'Performance at {lepochs} Epochs and {lbatch_size} Batch Size: {100*(1-collected_data[m, j, best_param_index, 0, 0]):.1f}% +- {100*collected_data[m, j, best_param_index, 0, 1]:.2f}')
            temp_var=float(100 * (1 - collected_data[m, j, best_param_index, 0, 0]))
            temp_group_score.append(temp_var)
            
        # print(temp_group_score)
        
    final_group_reco[model_name] = {
        "Male":temp_group_score[0],
        "Female":temp_group_score[1] ,
        "White":temp_group_score[2] ,
        "Black": temp_group_score[3]
    }


            


# In[ ]:


final_group_reco


# In[ ]:


all_state_recon


# In[ ]:


final_group_reco


# In[ ]:


filename_all_state_inversion = "inversion_normal_all_states_reconstruction.pickle"
filename_all_state_inversion_group = "inversion_normal_group_reconstruction.pickle"

# Open the file in binary write mode
with open(filename_all_state_inversion, 'wb') as file:
    pickle.dump(all_state_recon, file)

with open(filename_all_state_inversion_group, 'wb') as file:
    pickle.dump(final_group_reco, file)

print(f"Group reconstructed data has been stored in {filename_all_state_inversion}")
print(f"subGroup reconstructed data has been stored in {filename_all_state_inversion_group}")


# In[ ]:





# In[ ]:





# In[29]:


filename_all_state_inversion = "inversion_normal_all_states_reconstruction.pickle"
filename_all_state_inversion_group = "inversion_normal_group_reconstruction.pickle"

dp_train = "inversion_group_dp_fairness.pickle"
fair_train = "inversion_group_fair_fairness.pickle"
fairDP_train = "inversion_group_fairDp_fairness.pickle"


with open(filename_all_state_inversion, 'rb') as file:
    inversion_normal_all_state_reconstruction = pickle.load(file)
    
with open(filename_all_state_inversion_group, 'rb') as file:
    inversion_normal_group_reconstruction = pickle.load(file)


# with open(dp_train, 'rb') as file:
#     dp_train_loaded = pickle.load(file)
# with open(fair_train, 'rb') as file:
#     fair_train_loaded = pickle.load(file)
# with open(fairDP_train, 'rb') as file:
#     fairDP_train_loaded = pickle.load(file)




# In[33]:


inversion_normal_all_state_reconstruction


# In[32]:


inversion_normal_group_reconstruction


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Kacharo
# 

# In[ ]:





# In[ ]:


# exp1 = "tableak_FT_inversion_normal"
# CA_recn_gt = f'../{exp1}/50_clients_data/reconstr_and_GT/reconstructions_ground_truths_CA.pkl'
# dataset_CA = f'../{exp1}/50_clients_data/reconstr_and_GT/dataset_CA.pkl'
# tolerance_map_CA = f'../{exp1}/50_clients_data/reconstr_and_GT/tolerance_map_CA.pkl'


# -------- Inversion ------------------
# CA_recn_gt="50_clients_data/50_server_experiment_result/reconstr_and_GT_all_exp/inversion_dp/reconstructions_ground_truths_CA.pkl"
# dataset_CA="50_clients_data/50_server_experiment_result/reconstr_and_GT_all_exp/inversion_dp/dataset_CA.pkl"
# tolerance_map_CA="50_clients_data/50_server_experiment_result/reconstr_and_GT_all_exp/inversion_dp/tolerance_map_CA.pkl"


# CA_recn_gt="50_clients_data/50_server_experiment_result/reconstr_and_GT_all_exp/inversion_normal/reconstructions_ground_truths_AL.pkl"
# dataset_CA="50_clients_data/50_server_experiment_result/reconstr_and_GT_all_exp/inversion_normal/dataset_AL.pkl"
# tolerance_map_CA="50_clients_data/50_server_experiment_result/reconstr_and_GT_all_exp/inversion_normal/tolerance_map_AL.pkl"

# ------------ tableak -------------
# CA_recn_gt="50_clients_data/50_server_experiment_result/reconstr_and_GT_all_exp/tableak_normal/reconstructions_ground_truths_AL.pkl"
# dataset_CA="50_clients_data/50_server_experiment_result/reconstr_and_GT_all_exp/tableak_normal/dataset_AL.pkl"
# tolerance_map_CA="50_clients_data/50_server_experiment_result/reconstr_and_GT_all_exp/tableak_normal/tolerance_map_AL.pkl"


# with open(CA_recn_gt, 'rb') as file:
#     recn_gt = pickle.load(file)

# with open(dataset_CA, 'rb') as file:
#     dataset = pickle.load(file)

# with open(tolerance_map_CA, 'rb') as file:
#     tolerance_map = pickle.load(file)


# In[ ]:


def process_collector_data(collector_data):
    result = []
    for sublist in collector_data[0]:  # Access the first (and only) element of the outer list
        if not sublist:
            result.append(0)
        else:
            # Assuming there's only one value in each non-empty sublist
            result.append(sublist[0])
    return result


# In[6]:


def find_common_indices(*lists):
    
    # Extract sets of indices from all lists
    index_sets = [set(index for index, _ in lst) for lst in lists]
    
    # Find the intersection of all index sets
    common_indices = set.intersection(*index_sets)
    
    # Create a dictionary to store results
    result = {index: [] for index in common_indices}
    
    # Populate the result dictionary
    for lst in lists:
        for index, value in lst:
            if index in common_indices:
                result[index].append(value)
    
    return result


# In[7]:


gt_info=pd.read_csv('50_clients_data/state_sex_race_counts.csv')
gt_info.head()


# In[15]:


gt_info["Black_Count"].sort_values()


# In[17]:


gt_info[gt_info["State"]=="NY"]


# In[8]:


exp1 = "tableak_FT_inversion_normal"
config = configs[0]
final_grp={}
final_subgrp={}

print("inversion attack---2000 datapoints")
num_training_datapoints=2000

for model_name in client_models:
    print("model_name: ",model_name)
    temp_result={}
    
    recn_gt = f'../{exp1}/50_clients_data/reconstr_and_GT/reconstructions_ground_truths_{model_name}.pkl'
    dataset = f'../{exp1}/50_clients_data/reconstr_and_GT/dataset_{model_name}.pkl'
    tolerance_map = f'../{exp1}/50_clients_data/reconstr_and_GT/tolerance_map_{model_name}.pkl'

    with open(recn_gt, 'rb') as file:
        recn_gt = pickle.load(file)

    with open(dataset, 'rb') as file:
        dataset = pickle.load(file)
    
    with open(tolerance_map, 'rb') as file:
        tolerance_map = pickle.load(file)

    reconstructions = recn_gt['reconstructions']
    ground_truths = recn_gt['ground_truths']

    all_errors = []
    cat_errors = []
    cont_errors = []    
    
    collector_gen=[]
    collector_rac=[]
    collector_male=[]
    collector_female=[]
    collector_white=[]
    collector_black=[]

    
    for epoch_reconstruction, epoch_ground_truth in zip(reconstructions, ground_truths):
        for client_reconstruction, client_ground_truth in zip(epoch_reconstruction, epoch_ground_truth):
            if config['post_process_cont']:
                client_reconstruction = post_process_continuous(client_reconstruction, dataset=dataset)
            client_recon_projected, client_gt_projected = dataset.decode_batch(client_reconstruction, standardized=True), dataset.decode_batch(client_ground_truth, standardized=True)
            
            # print(type(client_recon_projected), type(client_gt_projected))
            # print(client_recon_projected.shape, client_gt_projected.shape)
    
            # _, batch_cost_all, batch_cost_cat, batch_cost_cont = match_reconstruction_ground_truth(client_gt_projected, client_recon_projected, tolerance_map)
    
            _, batch_cost_all, batch_cost_cat, batch_cost_cont,final_gen_recstr,final_rac_recstr,final_male_recstr,final_female_recstr,final_white_recstr,final_black_recstr = match_reconstruction_ground_truth_1(client_gt_projected, client_recon_projected, tolerance_map)
    
            all_errors.append(np.mean(batch_cost_all))
            cat_errors.append(np.mean(batch_cost_cat))
            cont_errors.append(np.mean(batch_cost_cont))
    
            collector_gen.append(final_gen_recstr)
            collector_rac.append(final_rac_recstr)
            collector_male.append(final_male_recstr)
            collector_female.append(final_female_recstr)
            collector_white.append(final_white_recstr)
            collector_black.append(final_black_recstr)
    

    male_rec_count=process_collector_data(collector_male).count(1)/num_training_datapoints
    female_rec_count=process_collector_data(collector_female).count(2)/num_training_datapoints
    white_rec_count=process_collector_data(collector_white).count(1)/num_training_datapoints
    black_rec_count=process_collector_data(collector_black).count(2)/num_training_datapoints


    gt_male_count=gt_info.loc[gt_info['State'] ==model_name, 'Male_Count'].values[0]
    gt_female_count=gt_info.loc[gt_info['State'] == model_name, 'Female_Count'].values[0]
    gt_white_count=gt_info.loc[gt_info['State'] == model_name, 'White_Count'].values[0]
    gt_black_count=gt_info.loc[gt_info['State'] == model_name, 'Black_Count'].values[0]


    temp_grp = {
        "Male": male_rec_count/gt_male_count,
        "Female":female_rec_count/gt_female_count ,
        "White":white_rec_count/gt_white_count,
        "Black":black_rec_count/gt_black_count}

    final_grp[model_name]=temp_grp


# In[ ]:


final_grp


# In[ ]:


filename_grp = "inversion_group_normal_reconstruction.pickle"
with open(filename_grp, 'wb') as file:
    pickle.dump(final_grp, file)

print(f"Group reconstructed data has been stored in {filename_grp}")


# In[ ]:


# filename_subgrrp = "inversion_subgroup_normal_reconstruction.pickle"

# with open(filename_subgrrp, 'wb') as file:
#     pickle.dump(final_subgrp, file)

# print(f"subGroup reconstructed data has been stored in {filename_subgrrp}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


# reconstructions = recn_gt['reconstructions']
# ground_truths = recn_gt['ground_truths']

# # 0 Inversion 52 Tableak
# config = configs[0]

# all_errors = []
# cat_errors = []
# cont_errors = []

# collector_gen=[]
# collector_rac=[]
# collector_male=[]
# collector_female=[]
# collector_white=[]
# collector_black=[]

# for epoch_reconstruction, epoch_ground_truth in zip(reconstructions, ground_truths):
#     for client_reconstruction, client_ground_truth in zip(epoch_reconstruction, epoch_ground_truth):
#         if config['post_process_cont']:
#             client_reconstruction = post_process_continuous(client_reconstruction, dataset=dataset)
#         client_recon_projected, client_gt_projected = dataset.decode_batch(client_reconstruction, standardized=True), dataset.decode_batch(client_ground_truth, standardized=True)
        
#         # print(type(client_recon_projected), type(client_gt_projected))
#         # print(client_recon_projected.shape, client_gt_projected.shape)

#         # _, batch_cost_all, batch_cost_cat, batch_cost_cont = match_reconstruction_ground_truth(client_gt_projected, client_recon_projected, tolerance_map)

        # _, batch_cost_all, batch_cost_cat, batch_cost_cont,final_gen_recstr,final_rac_recstr,final_male_recstr,final_female_recstr,final_white_recstr,final_black_recstr = match_reconstruction_ground_truth_1(client_gt_projected, client_recon_projected, tolerance_map)

#         all_errors.append(np.mean(batch_cost_all))
#         cat_errors.append(np.mean(batch_cost_cat))
#         cont_errors.append(np.mean(batch_cost_cont))

#         collector_gen.append(final_gen_recstr)
#         collector_rac.append(final_rac_recstr)
#         collector_male.append(final_male_recstr)
#         collector_female.append(final_female_recstr)
#         collector_white.append(final_white_recstr)
#         collector_black.append(final_black_recstr)




# In[ ]:





# In[28]:


total_num_points=2000
male_rec_count=process_collector_data(collector_male).count(1)/total_num_points
female_rec_count=process_collector_data(collector_female).count(2)/total_num_points

male_rec_count,female_rec_count

# print(f"Length of reconstr Male 0: {process_collector_data(collector_male).count(0)}")
# print(f"Length of reconstr Male 1: {process_collector_data(collector_male).count(1)}")
# print(f"Length of reconstr Female 0: {process_collector_data(collector_female).count(0)}")
# print(f"Length of reconstr Female 2: {process_collector_data(collector_female).count(2)}")


# In[ ]:





# In[9]:


white_rec_count=process_collector_data(collector_white).count(1)/total_num_points
black_rec_count=process_collector_data(collector_black).count(2)/total_num_points

white_rec_count,black_rec_count
# print(f"Length of reconstr White 0: {process_collector_data(collector_white).count(0)}")
# print(f"Length of reconstr Black 1: {process_collector_data(collector_white).count(1)}")
# print(f"Length of reconstr White 0: {process_collector_data(collector_black).count(0)}")
# print(f"Length of reconstr Black 2: {process_collector_data(collector_black).count(2)}")


# In[16]:





# In[24]:


gt_male_count=gt_info.loc[gt_info['State'] == 'CA', 'Male_Count'].values[0]
gt_female_count=gt_info.loc[gt_info['State'] == 'CA', 'Female_Count'].values[0]
gt_white_count=gt_info.loc[gt_info['State'] == 'CA', 'White_Count'].values[0]
gt_black_count=gt_info.loc[gt_info['State'] == 'CA', 'Black_Count'].values[0]


# In[25]:


white_rec_count/gt_white_count


# In[27]:


male_rec_count/gt_male_count


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# exp1 = "tableak_FT_inversion_normal"
# config = configs[0]
# final_grp={}
# final_subgrp={}

# print("inversion attack---2000 datapoints")
# num_training_datapoints=2000

# for model_name in client_models:
#     print("model_name: ",model_name)
#     temp_result={}
    
#     recn_gt = f'../{exp1}/50_clients_data/reconstr_and_GT/reconstructions_ground_truths_{model_name}.pkl'
#     dataset = f'../{exp1}/50_clients_data/reconstr_and_GT/dataset_{model_name}.pkl'
#     tolerance_map = f'../{exp1}/50_clients_data/reconstr_and_GT/tolerance_map_{model_name}.pkl'

#     with open(recn_gt, 'rb') as file:
#         recn_gt = pickle.load(file)

#     with open(dataset, 'rb') as file:
#         dataset = pickle.load(file)
    
#     with open(tolerance_map, 'rb') as file:
#         tolerance_map = pickle.load(file)

#     reconstructions = recn_gt['reconstructions']
#     ground_truths = recn_gt['ground_truths']

#     all_errors = []
#     cat_errors = []
#     cont_errors = []    
    
#     collector_gen=[]
#     collector_rac=[]
    
#     for epoch_reconstruction, epoch_ground_truth in zip(reconstructions, ground_truths):
#         for client_reconstruction, client_ground_truth in zip(epoch_reconstruction, epoch_ground_truth):
#             if config['post_process_cont']:
#                 client_reconstruction = post_process_continuous(client_reconstruction, dataset=dataset)
#             client_recon_projected, client_gt_projected = dataset.decode_batch(client_reconstruction, standardized=True), dataset.decode_batch(client_ground_truth, standardized=True)
            
#             # print(type(client_recon_projected), type(client_gt_projected))
#             # print(client_recon_projected.shape, client_gt_projected.shape)
    
#             # _, batch_cost_all, batch_cost_cat, batch_cost_cont = match_reconstruction_ground_truth(client_gt_projected, client_recon_projected, tolerance_map)
    
#             _, batch_cost_all, batch_cost_cat, batch_cost_cont,final_gen_recstr,final_rac_recstr,final_male_recstr,final_female_recstr = match_reconstruction_ground_truth_1(client_gt_projected, client_recon_projected, tolerance_map)
    
#             all_errors.append(np.mean(batch_cost_all))
#             cat_errors.append(np.mean(batch_cost_cat))
#             cont_errors.append(np.mean(batch_cost_cont))
    
#             collector_gen.append(final_gen_recstr)
#             collector_rac.append(final_rac_recstr)
#             collector_male.append(final_male_recstr)
#             collector_female.append(final_female_recstr)
    
#     processed_gen= process_collector_data(collector_gen)
#     processed_rac= process_collector_data(collector_rac)
    
#     temp_grp = {
#     "Male": math.ceil(processed_gen.count(1)/num_training_datapoints),
#     "Female": math.ceil(processed_gen.count(2)/num_training_datapoints),
#     "White":math.ceil(processed_rac.count(1)/num_training_datapoints),
#     "Black":math.ceil(processed_rac.count(2)/num_training_datapoints)
# }
#     gen_ones = [(i, val) for i, val in enumerate(processed_gen) if val == 1.0]
#     gen_twos = [(i, val) for i, val in enumerate(processed_gen) if val == 2.0]
#     rac_ones = [(i, val) for i, val in enumerate(processed_rac) if val == 1.0]
#     rac_twos = [(i, val) for i, val in enumerate(processed_rac) if val == 2.0]

#     wm = find_common_indices(gen_ones, rac_ones)
#     wf = find_common_indices(gen_twos, rac_ones)
#     bm = find_common_indices(gen_ones, rac_twos)
#     bf = find_common_indices(gen_twos, rac_twos)

#     temp_sub = {
#         "WM":math.ceil(len(wm)/num_training_datapoints),  # White Male
#         "WF": math.ceil(len(wf)/num_training_datapoints),  # White Female
#         "BM": math.ceil(len(bm)/num_training_datapoints),  # Black Male
#         "BF": math.ceil(len(bf)/num_training_datapoints)   # Black Female
#     }

#     final_grp[model_name]=temp_grp
#     final_subgrp[model_name]=temp_sub
             


# In[ ]:


final_grp


# In[9]:


final_subgrp


# In[10]:


filename_grp = "inversion_group_normal_reconstruction.pickle"
filename_subgrrp = "inversion_subgroup_normal_reconstruction.pickle"

# Open the file in binary write mode
with open(filename_grp, 'wb') as file:
    pickle.dump(final_grp, file)

with open(filename_subgrrp, 'wb') as file:
    pickle.dump(final_subgrp, file)

print(f"Group reconstructed data has been stored in {filename_grp}")
print(f"subGroup reconstructed data has been stored in {filename_subgrrp}")


# In[ ]:





# In[11]:


with open(filename_grp, 'rb') as file:
    normal_train_reconstruct_grp = pickle.load(file)


# In[13]:


with open(filename_subgrrp, 'rb') as file:
    normal_train_reconstruct_subgrp = pickle.load(file)


# In[7]:


1


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(normal_train_reconstruct_subgrp).T

# Create the box plot
plt.figure(figsize=(12, 6))
df.boxplot(column=['WM', 'WF', 'BM', 'BF'])

# Customize the plot
plt.title('Demographic Distribution Across States/Regions')
plt.ylabel('Count')
plt.xlabel('Demographic Categories')

# Add individual data points
for i, category in enumerate(['WM', 'WF', 'BM', 'BF'], 1):
    y = df[category]
    x = [i] * len(y)
    plt.scatter(x, y, alpha=0.5, color='red')

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[12]:


df = pd.DataFrame(normal_train_reconstruct_grp).T

# Create the box plot
plt.figure(figsize=(12, 6))
df.boxplot(column=['Male', 'Female', 'White', 'Black'])

# Customize the plot
plt.title('Demographic Distribution Across States/Regions')
plt.ylabel('Count')
plt.xlabel('Demographic Categories')

# Add individual data points
for i, category in enumerate(['Male', 'Female', 'White', 'Black'], 1):
    y = df[category]
    x = [i] * len(y)
    plt.scatter(x, y, alpha=0.5, color='red')

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


reconstructions = recn_gt['reconstructions']
ground_truths = recn_gt['ground_truths']

# 0 Inversion 52 Tableak
config = configs[0]


# In[21]:





# In[22]:


all_errors = []
cat_errors = []
cont_errors = []


collector_gen=[]
collector_rac=[]

for epoch_reconstruction, epoch_ground_truth in zip(reconstructions, ground_truths):
    for client_reconstruction, client_ground_truth in zip(epoch_reconstruction, epoch_ground_truth):
        if config['post_process_cont']:
            client_reconstruction = post_process_continuous(client_reconstruction, dataset=dataset)
        client_recon_projected, client_gt_projected = dataset.decode_batch(client_reconstruction, standardized=True), dataset.decode_batch(client_ground_truth, standardized=True)
        
        # print(type(client_recon_projected), type(client_gt_projected))
        # print(client_recon_projected.shape, client_gt_projected.shape)

        # _, batch_cost_all, batch_cost_cat, batch_cost_cont = match_reconstruction_ground_truth(client_gt_projected, client_recon_projected, tolerance_map)

        _, batch_cost_all, batch_cost_cat, batch_cost_cont,final_gen_recstr,final_rac_recstr = match_reconstruction_ground_truth_1(client_gt_projected, client_recon_projected, tolerance_map)

        all_errors.append(np.mean(batch_cost_all))
        cat_errors.append(np.mean(batch_cost_cat))
        cont_errors.append(np.mean(batch_cost_cont))

        collector_gen.append(final_gen_recstr)
        collector_rac.append(final_rac_recstr)


# In[23]:


len(collector_gen[0]),len(collector_rac[0])


# In[24]:





# In[25]:


processed_gen= process_collector_data(collector_gen)
processed_rac= process_collector_data(collector_rac)

# print(processed_gen)
print(f"Length of processed Gen: {len(processed_gen)}")
# print(processed_rac)
print(f"Length of processed Rac: {len(processed_rac)}")


# In[27]:


np.unique(processed_gen)


# In[26]:


np.unique(processed_rac)


# In[28]:


count_0_gen = processed_gen.count(0)
count_1_gen = processed_gen.count(1)
count_2_gen = processed_gen.count(2)
print("---Gender---")

print(f"Count of 0: {count_0_gen}")
print(f"Count of 1: {count_1_gen}")
print(f"Count of 2: {count_2_gen}")


# In[29]:


count_0_race = processed_rac.count(0)
count_1_race = processed_rac.count(1)
count_2_race = processed_rac.count(2)

print("---RACE---")
print(f"Count of 0: {count_0_race}")
print(f"Count of 1: {count_1_race}")
print(f"Count of 2: {count_2_race}")


# In[30]:


gen_ones = [(i, val) for i, val in enumerate(processed_gen) if val == 1.0]
gen_twos = [(i, val) for i, val in enumerate(processed_gen) if val == 2.0]
gen_ones, gen_twos


# In[31]:


rac_ones = [(i, val) for i, val in enumerate(processed_rac) if val == 1.0]
rac_twos = [(i, val) for i, val in enumerate(processed_rac) if val == 2.0]
rac_ones, rac_twos


# In[32]:


def find_common_indices(*lists):
    
    # Extract sets of indices from all lists
    index_sets = [set(index for index, _ in lst) for lst in lists]
    
    # Find the intersection of all index sets
    common_indices = set.intersection(*index_sets)
    
    # Create a dictionary to store results
    result = {index: [] for index in common_indices}
    
    # Populate the result dictionary
    for lst in lists:
        for index, value in lst:
            if index in common_indices:
                result[index].append(value)
    
    return result


common = find_common_indices(gen_ones, rac_ones)

print("Common indices and their values:")
# all_values = [value for values in common.values() for value in values]
# print(all_values)
total_count=0
for index, values in common.items():
    # print(f"Index {index}: {values}")
    total_count+=1

total_count


# In[ ]:


# 1 = Male, 0 = Female  
# 1 = White, 0 = Black  
# (real data: 1 male 2 female)
# (real data: 1 white 2 white)


# In[34]:


wm = find_common_indices(gen_ones, rac_ones)
wf = find_common_indices(gen_twos, rac_ones)
bm = find_common_indices(gen_ones, rac_twos)
bf = find_common_indices(gen_twos, rac_twos)



# In[36]:


import math


# In[37]:


print("inversion attack---2000 datapoints")
num_training_datapoints=2000
data = {
    "WM":math.ceil(len(wm)/num_training_datapoints),  # White Male
    "WF": math.ceil(len(wf)/num_training_datapoints),  # White Female
    "BM": math.ceil(len(bm)/num_training_datapoints),  # Black Male
    "BF": math.ceil(len(bf)/num_training_datapoints)   # Black Female
}

print("Results:")
for key, value in data.items():
    print(f"{key}: {value}")


# In[40]:


data2 = {
    "Male": processed_gen.count(1)/num_training_datapoints,
    "Female": processed_gen.count(2)/num_training_datapoints,
    "White": processed_rac.count(1)/num_training_datapoints,
    "Black": processed_rac.count(2)/num_training_datapoints
}



# In[41]:


print("Distribution:")
for category, count in data2.items():
    print(f"{category}: {count}")


# In[42]:


data2


# In[43]:


data


# In[ ]:





# In[ ]:




