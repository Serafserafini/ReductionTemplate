import numpy as np
import os 
import random
import pandas as pd
import time
from tqdm import tqdm
from template_csp.managetempcluster import generate_one_templateset, generate_one_pairset, graph_difference_std
import json

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


test_elements=['Be', 'B', 'N', 'Mg', 'O', 'Li', 'C', 'Na', 'Si', 'S', 'Cl', 'F', 'P', 'H', 'Al']
clusters = {}
with open('clusters.json', 'r') as f:
    clusters = json.load(f)
    


hyperparameters = {
    'ntemp_start' : 1,
    'ntemp_end' : 78,

    'comp' : 1,
    'lev_gen' : 0.8,
    'n_sets' : 5,
    'n_template' : 1,

    'id_set' : 1,
    'lev_red' : 0.9,
    'weight_formation_entalphy' : 1,
    'weight_occurrence' : 1,
    'weight_sg' : 0.001,

    'n_pairs' : 105,
    
}

random.seed(time.time())

for pesi in tqdm(range(20,101,20)):
    hyperparameters["n_pairs"] = pesi 
    
    # Number of possible couples
    n_possible_couples = 210
    if hyperparameters['comp'] == 1:
        n_possible_couples = 105


    dir_temp = f'./{hyperparameters["lev_red"]}_{hyperparameters["n_pairs"]}/'
    create_directory(dir_temp)

    import json
    with open(dir_temp + 'params.json', 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # Errore dei vari set con fissato numero di template estratti, su


# Errore dei vari set con fissato numero di template estratti, su ogni possible coppia
    dif_vec = np.zeros((hyperparameters['n_sets'], n_possible_couples)) 

    # Range in cui varia il numero di template estratti
    ntemp_studied = hyperparameters['ntemp_end'] - hyperparameters['ntemp_start']

    # Matrici per salvare i risultati al variare del numero di template estratti e su ogni coppia
    dif_mean = np.zeros((ntemp_studied, n_possible_couples))
    dif_std = np.zeros((ntemp_studied, n_possible_couples))

    # Vettori per i risultati globali
    means = np.zeros(ntemp_studied)
    stds = np.zeros(ntemp_studied)

    # Vettori per il numero di template rimanenti e la loro deviazione standard
    temp_red = np.zeros((2, ntemp_studied))

    for i in tqdm(range(hyperparameters['ntemp_start'],hyperparameters['ntemp_end'], 2)):

        hyperparameters['n_template'] = i

        # vettori per store di errore totale e numero di template rimanenti del singolo set
        errors = np.zeros(hyperparameters['n_sets'])
        tempafter = np.zeros(hyperparameters['n_sets'])
        
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

            # Salvataggio del template set su file
            create_directory(dir_temp+f'{template_set.num_template}')
            template_set.recap_tempset(dir_temp+f'{template_set.num_template}/TemplateSet_{k}')

            # Generazione del pair set
            reduction_set = generate_one_pairset(template_set, hyperparameters, test_elements)
            reduction_set.recap_relaxed(dir_temp+f'{i}/PairSet_{k}')

            # Salvataggio dei risultati per ogni set
            dif_vec[k]= np.array(reduction_set.error_single_composition(hyperparameters))
            errors[k] = reduction_set.total_error(hyperparameters) 
            tempafter[k] = len(reduction_set.reduced_set(hyperparameters))

        # Salvataggio dei risultati per ogni valore di template estratti
        # Numero di template rimanenti
        temp_red[0,i-hyperparameters['ntemp_start']] = np.mean(tempafter)
        temp_red[1,i-hyperparameters['ntemp_start']] = np.std(tempafter)
        # Errore medio con deviazione standard per ogni coppia
        dif_mean[i-hyperparameters['ntemp_start']] = np.mean(dif_vec, axis=0)
        dif_std[i-hyperparameters['ntemp_start']] = np.std(dif_vec, axis=0)
        graph_difference_std(dif_mean[i-hyperparameters['ntemp_start']] , dif_std[i-hyperparameters['ntemp_start']] , i, (i-hyperparameters['ntemp_start'])/ntemp_studied, dir_temp, hyperparameters, test_elements) 
        # Errore totale con deviazione standard
        means[i-hyperparameters['ntemp_start']] = np.mean(errors)
        stds[i-hyperparameters['ntemp_start']] = np.std(errors)

        pd_temp_red = pd.DataFrame(temp_red)
        pd_dif_mean = pd.DataFrame(dif_mean)
        pd_dif_std = pd.DataFrame(dif_std)

        df_tot = pd.DataFrame({'Means': means, 'Stds': stds})

        # Salvataggio dei risultati su file
        pd_dif_mean.to_csv(dir_temp+f'MeansEveryCouple.csv', header=None)
        pd_dif_std.to_csv(dir_temp+f'StdsEveryCouple.csv', header=None)
        df_tot.to_csv(dir_temp+f'TotalStatics.csv', header=None)
        pd_temp_red.to_csv(dir_temp+f'NumberTempRedu.csv', header=None)

