import numpy as np
import os 
import random
import pandas as pd
import time
from tqdm import tqdm
from template_csp.managetemp import generate_one_templateset, generate_one_pairset, graph_difference_std



def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


test_elements=['Be', 'B', 'N', 'Mg', 'O', 'Li', 'C', 'Na', 'Si', 'S', 'Cl', 'F', 'P', 'H', 'Al']

hyperparameters = {
    'ntemp_start' : 1,
    'ntemp_end' : 78,

    'comp' : 1,
    'lev_gen' : 0.8,
    'n_sets' : 10,
    'n_template' : 20,

    'id_set' : 1,
    'lev_red' : 0.0,
    'weight_formation_entalphy' : 1,
    'weight_occurrence' : 1,
    'weight_sg' : 0.001,

    'n_pairs' : 105,
    
}

random.seed(time.time())
en_err = np.zeros((2, 5, 11))
temp_red = np.zeros((2, 5, 11))

dir_temp = './LevRed_Npairs/'
create_directory(dir_temp)

for idx_lev, lev in tqdm(enumerate(range(82,85,2))):
    hyperparameters['lev_red'] = float(lev)/100

    for idx_npair, npair in tqdm(enumerate(range(20,101,20))):
        hyperparameters['n_pairs'] = npair
    
        errors = np.zeros(hyperparameters['n_sets'])
        tempafter = np.zeros(hyperparameters['n_sets'])
    
        with open('log.txt','a') as fstdout:
            fstdout.write('##################################################\n')
            fstdout.write(f'Generating template set with lev_red = {lev} and n_pairs = {npair}\n')
            fstdout.write('##################################################\n')
        
        create_directory(dir_temp+f'{hyperparameters["lev_red"]}_{hyperparameters["n_pairs"]}/')


        for k in range(hyperparameters['n_sets']):
            hyperparameters['lev_gen'] = 0.8
            hyperparameters['id_set'] = k

            template_set = generate_one_templateset(hyperparameters, test_elements)
            template_set.recap_tempset(dir_temp+f'{hyperparameters["lev_red"]}_{hyperparameters["n_pairs"]}/TemplateSet_{k}')

            reduction_set = generate_one_pairset(template_set, hyperparameters, test_elements)
            reduction_set.recap_relaxed(dir_temp+f'{hyperparameters["lev_red"]}_{hyperparameters["n_pairs"]}/PairSet_{k}')

            errors[k] = reduction_set.total_error(hyperparameters) 
            tempafter[k] = len(reduction_set.reduced_set(hyperparameters))


        temp_red[0,idx_npair,idx_lev] = np.mean(tempafter)
        temp_red[1,idx_npair,idx_lev] = np.std(tempafter)

        # Errore totale con deviazione standard
        en_err[0,idx_npair,idx_lev] = np.mean(errors)
        en_err[1,idx_npair,idx_lev] = np.std(errors)

EnErrMean = pd.DataFrame(en_err[0])
EnErrStd = pd.DataFrame(en_err[1])

NTempMean = pd.DataFrame(temp_red[0])
NTempStd = pd.DataFrame(temp_red[1])

EnErrMean.to_csv('./LevRed_Npairs/EnErrMean.csv', header=None)
EnErrStd.to_csv('LevRed_Npairs/EnErrStd.csv', header=None)

NTempMean.to_csv('./LevRed_Npairs/NTempMean.csv', header=None)
NTempStd.to_csv('LevRed_Npairs/NTempStd.csv', header=None)

