import os
import attacks
import numpy as np
import torch
from utils import match_reconstruction_ground_truth, Timer, post_process_continuous
from attacks import train_and_attack_fed_avg
from models import FullyConnected
from datasets import ADULT
import argparse
import pickle
from attacks import calculate_random_baseline

def calculate_fed_avg_local_dataset_inversion_performance(architecture_layout, dataset, max_client_dataset_size,
                                                          local_epochs, local_batch_sizes, epoch_prior_params,
                                                          tolerance_map, n_samples, config, max_n_cpus, first_cpu, device, state_name="AL"):
    
    collected_data = np.zeros((len(local_epochs), len(local_batch_sizes), len(epoch_prior_params), 3, 5))

    timer = Timer(len(local_epochs) * len(local_batch_sizes) * len(epoch_prior_params))
    
    with open(f'50_clients_data/reconstr_and_GT/dataset_{state_name}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    with open(f'50_clients_data/reconstr_and_GT/tolerance_map_{state_name}.pkl', 'wb') as f:
        pickle.dump(tolerance_map, f)

    for i, lepochs in enumerate(local_epochs):
        for j, lbatch_size in enumerate(local_batch_sizes):
            for k, epoch_prior_param in enumerate(epoch_prior_params):
                timer.start()                
                print(timer)
                # initialize the network (we do this everytime, giving us independent experiments)

                net = FullyConnected(dataset.num_features, architecture_layout)
                # pre_trained_model_path="50_clients_data/clients_trained_model/pre_trained_model.pth"
                # net.load_state_dict(torch.load(pre_trained_model_path))

                # only include the epoch matching prior if the corresponding parameter is non-zero
                epoch_matching_prior = (epoch_prior_param, config['epoch_matching_prior']) if epoch_prior_param > 0. else None

                # train with fedavg and attack
                _, _, reconstructions, ground_truths = train_and_attack_fed_avg(
                    net=net,
                    n_clients=n_samples,
                    n_global_epochs=config['n_global_epochs'],
                    n_local_epochs=lepochs,
                    local_batch_size=lbatch_size,
                    lr=config['lr'],
                    dataset=dataset,
                    shuffle=config['shuffle'],
                    attacked_clients=config['attacked_clients'],
                    attack_iterations=config['attack_iterations'],
                    reconstruction_loss=config['reconstruction_loss'],
                    priors=config['priors'],
                    epoch_matching_prior=epoch_matching_prior,
                    post_selection=config['post_selection'],
                    attack_learning_rate=config['attack_learning_rate'],
                    return_all=config['return_all'],
                    pooling=config['pooling'],
                    perfect_pooling=config['perfect_pooling'],
                    initialization_mode=config['initialization_mode'],
                    softmax_trick=config['softmax_trick'],
                    gumbel_softmax_trick=config['gumbel_softmax_trick'],
                    sigmoid_trick=config['sigmoid_trick'],
                    temperature_mode=config['temperature_mode'],
                    sign_trick=config['sign_trick'],
                    fish_for_features=None,
                    max_n_cpus=max_n_cpus,
                    first_cpu=first_cpu,
                    device=device,
                    verbose=False,
                    max_client_dataset_size=max_client_dataset_size,
                    parallelized=False,
                    state_name=state_name
                )

                # calculate the inversion error
                all_errors = []
                cat_errors = []
                cont_errors = []

                # Save reconstructions and ground_truths to a pickle file
                with open(f'50_clients_data/reconstr_and_GT/reconstructions_ground_truths_{state_name}.pkl', 'wb') as f:
                    pickle.dump({'reconstructions': reconstructions, 'ground_truths': ground_truths}, f)

                print("reconstructions_and_ground_truths is dumped")

                # print("reconstructions Shape  ",reconstructions[0][0].shape)
                # # print("reconstructions   ",reconstructions)
                # print("ground_truth Shape ", ground_truths[0][0].shape)
                # print("ground_truths   ", ground_truths)

                for epoch_reconstruction, epoch_ground_truth in zip(reconstructions, ground_truths):
                    for client_reconstruction, client_ground_truth in zip(epoch_reconstruction, epoch_ground_truth):
                        if config['post_process_cont']:
                            client_reconstruction = post_process_continuous(client_reconstruction, dataset=dataset)
                        client_recon_projected, client_gt_projected = dataset.decode_batch(client_reconstruction, standardized=True), dataset.decode_batch(client_ground_truth, standardized=True)
                        _, batch_cost_all, batch_cost_cat, batch_cost_cont = match_reconstruction_ground_truth(client_gt_projected, client_recon_projected, tolerance_map)
                        all_errors.append(np.mean(batch_cost_all))
                        cat_errors.append(np.mean(batch_cost_cat))
                        cont_errors.append(np.mean(batch_cost_cont))

                collected_data[i, j, k, 0] = np.mean(all_errors), np.std(all_errors), np.median(all_errors), np.min(all_errors), np.max(all_errors)
                collected_data[i, j, k, 1] = np.mean(cat_errors), np.std(cat_errors), np.median(cat_errors), np.min(cat_errors), np.max(cat_errors)
                collected_data[i, j, k, 2] = np.mean(cont_errors), np.std(cont_errors), np.median(cont_errors), np.min(cont_errors), np.max(cont_errors)

                timer.end()

            best_param_index = np.argmin(collected_data[i, j, :, 0, 0]).item()

            print(f'Performance at {lepochs} Epochs and {lbatch_size} Batch Size: {100*(1-collected_data[i, j, best_param_index, 0, 0]):.1f}% +- {100*collected_data[i, j, best_param_index, 0, 1]:.2f}')
            
            display_map = {
                'mean': 0,
                'std': 1,
                'median': 2,
                'min': 3,
                'max': 4
            }
            display = 'mean'
            random_baseline = calculate_random_baseline(dataset=dataset, recover_batch_sizes=[lbatch_size],
                                                        tolerance_map=tolerance_map, n_samples=n_samples)
            batch_sizes = [lbatch_size]
            # print("random acc:  ",random_baseline)
            for l, batch_size in enumerate(batch_sizes):
                print("random_baseline", (np.around(100 - 100*random_baseline[l, 0, display_map[display]], 1), np.around(100*random_baseline[l, 0, 1], 1)))
    return collected_data


def main(args):
    # print(args)

    datasets = {
        'ADULT': ADULT,
    }

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

    # ------------ PARAMETERS ------------ #

    architecture_layout = [100, 100, 2]  # network architecture (fully connected)
    max_client_dataset_size = 32
    local_epochs =[5] #[1, 5, 10] 
    # IMO: good batch to reconstruct the data-: max/local = should be smaller  (512/64)(128/16)
    local_batch_sizes = [8] #[32, 16, 8]
    epoch_prior_params =[0.01] #[0.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    tol = 0.319

    # ------------ END ------------ #

    # get the configuration
    config = configs[args.experiment]
    # prepare the dataset
    # print("start")
    dataset = datasets[args.dataset](device=args.device, random_state=args.random_seed,name_state=args.name_state)

    # print("end:: ", datasets[args.dataset])

    dataset.standardize()
    tolerance_map = dataset.create_tolerance_map(tol=tol)
    

    # set the random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)


    # ------------ INVERSION EXPERIMENT ------------ #
    base_path = f'experiment_data/fedavg_experiments/{args.dataset}/experiment_{args.experiment}'
    os.makedirs(base_path, exist_ok=True)
    specific_file_path = base_path + f'/inversion_data_all_{args.experiment}_{args.dataset}_{args.n_samples}_{epoch_prior_params}_{tol}_{args.random_seed}_{args.name_state}.npy'

    if os.path.isfile(specific_file_path) and not args.force:
        print('This experiment has already been conducted')
    else:
        inversion_data = calculate_fed_avg_local_dataset_inversion_performance(
            architecture_layout=architecture_layout,
            dataset=dataset,
            max_client_dataset_size=max_client_dataset_size,
            local_epochs=local_epochs,
            local_batch_sizes=local_batch_sizes,
            epoch_prior_params=epoch_prior_params,
            tolerance_map=tolerance_map,
            n_samples=args.n_samples,
            config=config,
            max_n_cpus=args.max_n_cpus,
            first_cpu=args.first_cpu,
            device=args.device,
            state_name=args.name_state
        )
        np.save(specific_file_path, inversion_data)
    print('Complete                           ')
    print('==================================================================')
    print('==================================================================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('run_inversion_parser')
    parser.add_argument('--dataset', type=str, default='ADULT', help='Select the dataset')
    # 52 means Tableak , 0 means Inverting Gradients
    parser.add_argument('--experiment', type=int, default=0, help='Select the experiment you wish to run') 
    parser.add_argument('--name_state', type=str, default='AL', help='State Code')
    parser.add_argument('--n_samples', type=int, default=1,help='Set the number of MC samples taken for each experiment')
    parser.add_argument('--random_seed', type=int, default=2, help='Set the random state for reproducibility')
    parser.add_argument('--max_n_cpus', type=int, default=4, help='The number of available cpus')
    parser.add_argument('--first_cpu', type=int, default=0, help='The first cpu in the pool')
    parser.add_argument('--force', action='store_true', help='If set to true, this will force the program to redo a given experiment')
    parser.add_argument('--device', type=str, default='cpu', help='Select the device to run the program on')
    in_args = parser.parse_args()
    main(in_args)
