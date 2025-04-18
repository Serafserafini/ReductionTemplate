import numpy as np
import os 
import random
import pandas as pd
import time
from tqdm import tqdm
from template_csp.managetemp_withdict import generate_one_templateset, generate_one_pairset
import template_csp.managetemp_withdict as mte
from template_csp.distances import levensthein_distance
import json
import argparse

# os.environ["OMP_NUM_THREADS"] = "128"  # Per OpenMP
# os.environ["OPENBLAS_NUM_THREADS"] = "128"  # Per NumPy
# os.environ["MKL_NUM_THREADS"] = "128"  # Per Intel MKL
# os.environ["NUMEXPR_NUM_THREADS"] = "128"  # Per NumExpr

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Argomenti da riga di comando
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--comp', type=int, default=1, help='Composition of the compounds')
parser.add_argument('-i', '--ntempinit', type=int, default=1, help='Number of templates to start with')
parser.add_argument('-f', '--ntempfinal', type=int, default=50, help='Number of templates to end with')
parser.add_argument('-s', '--sets', type=int, default=20, help='Number of sets to generate')
# parser.add_argument('-j', '--jobid', type=int, default=1, help='Job ID')
args = parser.parse_args()


comp = args.comp
ntemp_start = args.ntempinit
ntemp_end = args.ntempfinal
n_sets = args.sets
job_id = args.jobid

test_elements=['Be', 'B', 'N', 'Mg', 'O', 'Li', 'C', 'Na', 'Si', 'S', 'Cl', 'F', 'P', 'H', 'Al']

hyperparameters = {
    'ntemp_start' : ntemp_start,
    'ntemp_end' : ntemp_end,

    'comp' : comp,
    'lev_gen' : 0.0,
    'lev_gen_initial' : 0.0,
    'step' : 0.1,
    'n_sets' : n_sets,
    'n_template' : 1,

    'id_set' : 1,
    'lev_red' : 0.9,
    'weight_formation_entalphy' : 1,
    'weight_occurrence' : 1,
    'weight_sg' : 0.001,
    'n_final_templates' : 1,

    'n_pairs' : 210,    
    # 'job_id' : job_id
}
random.seed(time.time())
# ntem = [86, 157, 92, 79]

complist = [1,2,3,4]
final_size = [10,20,15,15]

for comp in complist:
    hyperparameters['comp'] = comp
    hyperparameters['n_final_templates'] = final_size[complist.index(comp)]
    if hyperparameters['comp'] == 1:
        n_possible_couples = 105
        hyperparameters['n_pairs'] = 105
    else:
        n_possible_couples = 210

    dir_temp = f'./{hyperparameters["comp"]}/'
    create_directory(dir_temp)

    with open(dir_temp + 'params.json', 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # Range in cui varia il numero di template estratti
    ntemp_studied = hyperparameters['ntemp_end'] - hyperparameters['ntemp_start'] + 1

    # Vettori per i risultati globali
    means = np.zeros(ntemp_studied)
    stds = np.zeros(ntemp_studied)

    # tempmeans = np.zeros(ntemp_studied)
    # tempstds = np.zeros(ntemp_studied)

    means_bef = np.zeros(ntemp_studied)
    stds_bef = np.zeros(ntemp_studied)

    for i in tqdm(range(hyperparameters['ntemp_start'] , hyperparameters['ntemp_end'] + 1)):

        hyperparameters['n_template'] = i

        # vettori per store di errore totale e numero di template rimanenti del singolo set
        errors = np.zeros(hyperparameters['n_sets'])
        # n_templates = np.zeros(hyperparameters['n_sets'])
        errors_bef = np.zeros(hyperparameters['n_sets'])
        
        # with open(f'log{hyperparameters["job_id"]}.txt','a') as fstdout:
        with open(f'log.txt','a') as fstdout:
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
            errors[k] = reduced_set.total_error()
            # n_templates[k] = len(reduced_set.reduced_set())
            errors_bef[k] = template_set.err_before()


        # Errore totale con deviazione standard
        means[i - hyperparameters['ntemp_start'] ] = np.mean(errors)
        stds[i - hyperparameters['ntemp_start'] ] = np.std(errors)
        
        with open(dir_temp+f'TotalStatics.csv', 'a') as f:
            f.write(f'{i-1}, {means[i - hyperparameters['ntemp_start'] ]}, {stds[i - hyperparameters['ntemp_start'] ]}\n')

        # Errore totale con deviazione standard prima della riduzione
        means_bef[i - hyperparameters['ntemp_start']] = np.mean(errors_bef)
        stds_bef[i - hyperparameters ['ntemp_start']] = np.std(errors_bef)

        with open(dir_temp+f'TotalStaticsBefore.csv', 'a') as f:
            f.write(f'{i-1}, {means_bef[i - hyperparameters['ntemp_start'] ]}, {stds_bef[i - hyperparameters['ntemp_start'] ]}\n')




