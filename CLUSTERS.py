import numpy as np
import os 
import random
import pandas as pd
import time
from tqdm import tqdm
from template_csp.managetemp import generate_one_templateset, generate_one_pairset, graph_difference_std
from template_csp.distances import dist1,dist2,dist3
import json

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

test_elements=['Be', 'B', 'N', 'Mg', 'O', 'Li', 'C', 'Na', 'Si', 'S', 'Cl', 'F', 'P', 'H', 'Al']

hyperparameters = {
    'ntemp_start' : 1,
    'ntemp_end' : 30,

    'comp' : 1,
    'lev_gen' : 0.8,
    'n_sets' : 50,
    'n_template' : 1,

    'id_set' : 1,
    'lev_red' : 0.05,
    'weight_formation_entalphy' : 1,
    'weight_occurrence' : 1,
    'weight_sg' : 0.001,

    'n_pairs' : 40,
}

random.seed(time.time())

n_possible_couples = 210
if hyperparameters['comp'] == 1:
    n_possible_couples = 105

for cluster_idx in range(8,9):
    clusters = {}
    with open(f'ClustersDict/Clusters{cluster_idx}.json', 'r') as f:
        clusters = json.load(f)


    dir_temp = f'./CLUSTER/{cluster_idx}_Clusters/'
    create_directory(dir_temp)

    import json
    with open(dir_temp + 'params.json', 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # Range in cui varia il numero di template estratti
    ntemp_studied = hyperparameters['ntemp_end'] - hyperparameters['ntemp_start']

    # Vettori per i risultati globali
    means = np.zeros(ntemp_studied)
    stds = np.zeros(ntemp_studied)

    # Vettori per il numero di template rimanenti e la loro deviazione standard
    temp_red = np.zeros((2, ntemp_studied))

    means_before = np.zeros(ntemp_studied)
    stds_before = np.zeros(ntemp_studied)

    for i in tqdm(range(hyperparameters['ntemp_start'],hyperparameters['ntemp_end'], 1)):

        hyperparameters['n_template'] = i

        # vettori per store di errore totale e numero di template rimanenti del singolo set
        errors = np.zeros(hyperparameters['n_sets'])
        tempafter = np.zeros(hyperparameters['n_sets'])
        errbefore = np.zeros(hyperparameters['n_sets'])
        
        with open('log.txt','a') as fstdout:
            fstdout.write('##################################################\n')
            fstdout.write(f'Generating template set with {i} templates\n')
            fstdout.write('##################################################\n')

        for k in range(hyperparameters['n_sets']):
            # Reset delle variabili
            hyperparameters['lev_gen'] = 0.8
            hyperparameters['id_set'] = k

            # Generazione del template set inziale
            template_set = generate_one_templateset(hyperparameters, test_elements, clusters)
            errbefore[k] = template_set.err_before()

            # Salvataggio del template set su file
            create_directory(dir_temp+f'{template_set.num_template}')
            template_set.recap_tempset(dir_temp+f'{template_set.num_template}/TemplateSet_{k}')

            # Generazione del pair set
            reduction_set = generate_one_pairset(template_set, hyperparameters, test_elements, dist2)
            reduction_set.recap_relaxed(dir_temp+f'{i}/PairSet_{k}')

            # Salvataggio dei risultati per ogni set
            errors[k] = reduction_set.total_error(hyperparameters) 
            tempafter[k] = len(reduction_set.reduced_set(hyperparameters))

        # Salvataggio dei risultati per ogni valore di template estratti
        # Numero di template rimanenti
        temp_red[0,i-hyperparameters['ntemp_start']] = np.mean(tempafter)
        temp_red[1,i-hyperparameters['ntemp_start']] = np.std(tempafter)

        # Errore totale con deviazione standard
        means[i-hyperparameters['ntemp_start']] = np.mean(errors)
        stds[i-hyperparameters['ntemp_start']] = np.std(errors)

        # Errore totale prima della riduzione
        means_before[i-hyperparameters['ntemp_start']] = np.mean(errbefore)
        stds_before[i-hyperparameters['ntemp_start']] = np.std(errbefore)

        # Salvataggio dei risultati su file
        pd_temp_red = pd.DataFrame(temp_red)
        df_tot = pd.DataFrame({'Means': means, 'Stds': stds})
        df_tot_before = pd.DataFrame({'Means': means_before, 'Stds': stds_before})

        # Salvataggio dei risultati su file
        df_tot_before.to_csv(dir_temp+f'TotalStaticsBefore.csv', header=None)
        df_tot.to_csv(dir_temp+f'TotalStatics.csv', header=None)
        pd_temp_red.to_csv(dir_temp+f'NumberTempRedu.csv', header=None)

