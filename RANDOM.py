import numpy as np
import os 
import random
import pandas as pd
import time
from tqdm import tqdm
from template_csp.managetemp_withdict import generate_one_templateset, generate_one_pairset, graph_difference_std
import template_csp.managetemp_withdict as mte
from template_csp.distances import levensthein_distance, dist1, dist2, dist3
import json


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


test_elements=['Be', 'B', 'N', 'Mg', 'O', 'Li', 'C', 'Na', 'Si', 'S', 'Cl', 'F', 'P', 'H', 'Al']

hyperparameters = {
    'ntemp_start' : 1,
    'ntemp_end' : 30,

    'comp' : 2,
    'lev_gen' : 0.8,
    'lev_gen_initial' : 0.8,
    'step' : 0.1,
    'n_sets' : 5,
    'n_template' : 1,

    'id_set' : 1,
    'lev_red' : 0.9,
    'weight_formation_entalphy' : 1,
    'weight_occurrence' : 1,
    'weight_sg' : 0.001,

    'n_pairs' : 210,    
}
random.seed(time.time())

if hyperparameters['comp'] == 1:
    n_possible_couples = 105
else:
    n_possible_couples = 210

dir_temp = f'./{hyperparameters['comp']}/'
create_directory(dir_temp)

import json
with open(dir_temp + 'params.json', 'w') as f:
    json.dump(hyperparameters, f, indent=4)

# Range in cui varia il numero di template estratti
ntemp_studied = hyperparameters['ntemp_end'] - hyperparameters['ntemp_start']

# Vettori per i risultati globali
means = np.zeros(ntemp_studied)
stds = np.zeros(ntemp_studied)

tempmeans = np.zeros(ntemp_studied)
tempstds = np.zeros(ntemp_studied)

for i in tqdm(range(hyperparameters['ntemp_start'],hyperparameters['ntemp_end'], 1)):

    hyperparameters['n_template'] = i

    # vettori per store di errore totale e numero di template rimanenti del singolo set
    errors = np.zeros(hyperparameters['n_sets'])
    n_templates = np.zeros(hyperparameters['n_sets'])
    
    with open('log.txt','a') as fstdout:
        fstdout.write('##################################################\n')
        fstdout.write(f'Generating template set with {i} templates\n')
        fstdout.write('##################################################\n')

    create_directory(dir_temp+f'{hyperparameters["n_template"]}/')

    for k in range(hyperparameters['n_sets']):
        # Reset delle variabili
        hyperparameters['lev_gen'] = hyperparameters['lev_gen_initial']
        hyperparameters['id_set'] = k

        # Generazione del template set inziale
        template_set = generate_one_templateset(hyperparameters, test_elements)
        template_set.recap_tempset(dir_temp+f'{hyperparameters["n_template"]}/TemplateSet_{k}')

        reduced_set = generate_one_pairset(template_set, hyperparameters, test_elements)
        reduced_set.recap_relaxed(dir_temp+f'{hyperparameters["n_template"]}/PairSet_{k}')

        # Salvataggio del template set su file
    
        # Salvataggio dei risultati per ogni set
        errors[k] = reduced_set.total_error(hyperparameters)
        n_templates[k] = len(reduced_set.reduced_set(hyperparameters))



    # Errore totale con deviazione standard
    means[i-hyperparameters['ntemp_start']] = np.mean(errors)
    stds[i-hyperparameters['ntemp_start']] = np.std(errors)

    # Numero di template rimanenti con deviazione standard
    tempmeans[i-hyperparameters['ntemp_start']] = np.mean(n_templates)
    tempstds[i-hyperparameters['ntemp_start']] = np.std(n_templates)
    
    df_tot = pd.DataFrame({'Means': means, 'Stds': stds})

    df_temp = pd.DataFrame({'Means': tempmeans, 'Stds': tempstds})

    # Salvataggio dei risultati su file
    df_tot.to_csv(dir_temp+f'TotalStatics.csv', header=None)
    df_temp.to_csv(dir_temp+f'NumberTempRedu.csv', header=None)