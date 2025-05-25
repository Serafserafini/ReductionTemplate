from template_csp import managetemp as mte 
import json
from tqdm import tqdm
import numpy as np

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
# data1 = {}

complist = [1,2,3,4,6]
# temp_list = [[20,30,40], [40,50,60], [30,40,50], [30,40,50], [20,30,40]] 

temp_final = [10, 20, 15, 15, 10]

temp_list = [[30],[50],[40],[40],[30]]
total_pairs_list = None
n_set = 20
n_try_pairs = 30

crit_pairs = './CriticalPairs.json'
mother_dir = './HvsINIT/'

for comp, temp_init_list in tqdm(zip(complist, temp_list)):
    
    hyperparameters["comp"] = comp
    
    if total_pairs_list:
        total_pairs = total_pairs_list[complist.index(comp)]
    else:
        if comp == 1:
            total_pairs = 105
        else:
            total_pairs = 210

    data[comp] = {'ErrBef': {}, 'ErrAft': {}, 'StdAft':{} }
    # data1[comp] = {}

    hyperparameters['n_final_templates'] = temp_final[complist.index(comp)]

    for idx_temp, temp_init in enumerate(temp_init_list):

        data[comp]['ErrBef'][temp_init] = 0
        data[comp]['ErrAft'][temp_init] = {}
        data[comp]['StdAft'][temp_init] = {}
        # data1[comp][temp_init] = {}

        for id_set in range(n_set):

            init_set = mte.InitialSet(test_elements, hyperparameters, mother_dir+f'{comp}/{temp_init}/InitialSet_{id_set}')
            
            data[comp]['ErrBef'][temp_init] += init_set.difference_from_uspex()/n_set
            
            for idx_npair, npair in enumerate(range(1, total_pairs, 5)):

                if npair not in data[comp]['ErrAft'][temp_init].keys():
                    data[comp]['ErrAft'][temp_init][npair] = []
  
                    # data1[comp][temp_init][npair] = 0

                hyperparameters["n_pairs"] = npair
                
                for try_pairs in range(n_try_pairs):
                    final_set = mte.generate_final_set(init_set, hyperparameters, test_elements)
                    data[comp]['ErrAft'][temp_init][npair].append(final_set.difference_from_uspex())

                    final_set.recap(f'{npair}')
                    
                    # data1[comp][temp_init][npair] += float(final_set.count_crit_in_validation)/(n_set*n_try_pairs)
        
            
            npair = total_pairs
            hyperparameters["n_pairs"] = npair

            final_set = mte.generate_final_set(init_set, hyperparameters, test_elements)
            if npair not in data[comp]['ErrAft'][temp_init].keys():
                data[comp]['ErrAft'][temp_init][npair] = []
                # data1[comp][temp_init][npair] = 0
            
            # data1[comp][temp_init][npair] += float(final_set.count_crit_in_validation)/n_set
            data[comp]['ErrAft'][temp_init][npair].append(final_set.difference_from_uspex())

        for npair in data[comp]['ErrAft'][temp_init].keys():
            data[comp]['StdAft'][temp_init][npair] = np.array(data[comp]['ErrAft'][temp_init][npair]).std()
            data[comp]['ErrAft'][temp_init][npair] = np.array(data[comp]['ErrAft'][temp_init][npair]).mean()

        with open('./NumberOfPairs_WeightSquared.json', 'w') as f:
            json.dump(data, f, indent=4)


            # with open('./UsedCritPairs.json', 'w') as f:
                # json.dump(data1, f, indent=4) 
