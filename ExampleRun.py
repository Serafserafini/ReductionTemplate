from template_csp import manage_alltemp as mte 
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

    "n_final_templates" : 1,
    #"critical_pairs" : './CriticalPairs.json'
}

data = {}
data1 = {}

complist = [1,2,3,4,6]
temp_list = [[20,30,40], [40,50,60], [30,40,50], [30,40,50], [20,30,40]] 
temp_final = [10, 20, 15, 15, 10]
#total_pairs_list = [95, 180, 175, 171, 210]
total_pairs_list = None
n_set = 20
n_try_pairs = 30

crit_pairs = './CriticalPairs.json'

for comp, temp_init_list in tqdm(zip(complist, temp_list)):
    
    if comp == 6:
        continue

    hyperparameters["comp"] = comp
    
    if total_pairs_list:
        total_pairs = total_pairs_list[complist.index(comp)]
    else:
        if comp == 1:
            total_pairs = 105
        else:
            total_pairs = 210

    data[comp] = {'ErrBef': {}, 'ErrAft': {}}
    data1[comp] = {}

    hyperparameters['n_final_templates'] = temp_final[complist.index(comp)]

    for idx_temp, temp_init in enumerate(temp_init_list):

        data[comp]['ErrBef'][temp_init] = 0
        data[comp]['ErrAft'][temp_init] = {}
        data1[comp][temp_init] = {}    

        for id_set in range(n_set):

            init_set = mte.generate_initial_set(hyperparameters=hyperparameters, test_elements=test_elements)
            init_set.recap()
            
            data[comp]['ErrBef'][temp_init] += init_set.difference_from_uspex()/n_set
            
            for idx_npair, npair in enumerate(range(1, total_pairs, 5)):

                if npair not in data[comp]['ErrAft'][temp_init].keys():
                    data[comp]['ErrAft'][temp_init][npair] = 0
                    data1[comp][temp_init][npair] = 0

                hyperparameters["n_pairs"] = npair
                
                for try_pairs in range(n_try_pairs):
                    final_set = mte.generate_final_set(init_set, hyperparameters=hyperparameters, test_elements=test_elements, file_crit_pairs = crit_pairs)
                    
                    data[comp]['ErrAft'][temp_init][npair] += final_set.difference_from_uspex()/(n_set*n_try_pairs)
                    data1[comp][temp_init][npair] += float(final_set.count_crit_in_validation)/(n_set*n_try_pairs)
                    
                    final_set.recap()
            
            npair = total_pairs
            hyperparameters["n_pairs"] = npair

            final_set = mte.generate_final_set(init_set, hyperparameters=hyperparameters, test_elements=test_elements, file_crit_pairs = crit_pairs)
            if npair not in data[comp]['ErrAft'][temp_init].keys():
                data[comp]['ErrAft'][temp_init][npair] = 0
                data1[comp][temp_init][npair] = 0
            
            data1[comp][temp_init][npair] += float(final_set.count_crit_in_validation)/n_set
            data[comp]['ErrAft'][temp_init][npair] += final_set.difference_from_uspex()/n_set

            with open('./NumberOfPairs_Continuum.json', 'w') as f:
                json.dump(data, f, indent=4)
                
            with open('./UsedCritPairs.json', 'w') as f:
                json.dump(data1, f, indent=4) 
