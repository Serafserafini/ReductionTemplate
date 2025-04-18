from template_csp import manage as mte 
import json
from tqdm import tqdm

test_elements = ['Be', 'B', 'N', 'Mg', 'O', 'Li', 'C', 'Na', 'Si', 'S', 'Cl', 'F', 'P', 'H', 'Al']

hyperparameters = { 
    "weight_occurrence" : 1,
    "weight_sg" : 0.001,
    "weight_formation_entalphy" : 1, 
    "comp" : 1,
    "lev_red" : 0.9,
    "n_pairs" : 105,

    "n_final_templates" : 1
}

data = {}

complist = [1,2,3,4,6]
temp_list = [[20,30,40], [40,50,60], [30,40,50], [30,40,50], [20,30,40]] 
temp_final = [10, 20, 15, 15, 10]

n_set = 20
n_try_pairs = 30


mother_dir = './HvsINIT/'

for comp, temp_init_list in tqdm(zip(complist, temp_list)):

    hyperparameters["comp"] = comp

    if comp == 1:
        total_pairs = 105
    else:
        total_pairs = 210

    data[comp] = {'ErrBef': {}, 'ErrAft': {}}

    hyperparameters['n_final_templates'] = temp_final[complist.index(comp)]

    for idx_temp, temp_init in enumerate(temp_init_list):

        data[comp]['ErrBef'][temp_init] = 0
        data[comp]['ErrAft'][temp_init] = {}

        for id_set in range(n_set):

            init_set = mte.TemplateSet(test_elements=test_elements, hyperparameters=hyperparameters, restart_file=mother_dir+f'{comp}/{temp_init}/TemplateSet_{id_set}', comp=comp)
            
            data[comp]['ErrBef'][temp_init] += init_set.err_before()/n_set
            
            for idx_npair, npair in enumerate(range(1, total_pairs, 2)):

                if npair not in data[comp]['ErrAft'][temp_init].keys():
                    data[comp]['ErrAft'][temp_init][npair] = 0
            
                hyperparameters["n_pairs"] = npair
                
                for try_pairs in range(n_try_pairs):
                    final_set = mte.generate_one_pairset(init_set, hyperparameters=hyperparameters, test_elements=test_elements)
                    data[comp]['ErrAft'][temp_init][npair] += final_set.total_error()/(n_set*n_try_pairs)
                
        
            
            npair = 105 if comp == 1 else 210
            hyperparameters["n_pairs"] = npair

            final_set = mte.generate_one_pairset(init_set, hyperparameters=hyperparameters, test_elements=test_elements)
            if npair not in data[comp]['ErrAft'][temp_init].keys():
                data[comp]['ErrAft'][temp_init][npair] = 0
            data[comp]['ErrAft'][temp_init][npair] += final_set.total_error()/n_set

        with open('./NumberOfPairs_Continuum.json', 'w') as f:
            json.dump(data, f, indent=4)
                