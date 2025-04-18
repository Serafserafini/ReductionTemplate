import numpy as np
import os 
import os.path as op
import re
import random
import pandas as pd
import json


from pymatgen.core import Element

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from template_csp.manageuspex import read_individuals, best_structures
from template_csp import manageqe as mqe
from template_csp.distances import levensthein_distance
from template_csp import cif2qe as c2q


#Remove the file
def remove_file(file):
    if os.path.exists(file):
        os.remove(file)
    return

#Create the directory
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


class TemplateSet:
    def __init__(self, test_elements, hyperparameters, restart_file = None, comp = 1, mother_dir = './TEMPLATES/', flag_one_temp_per_pair = False, flag_all_temp = False) -> None:
        self.dir_all_Individuals = op.join(mother_dir , f'SETUP_COMP{comp}', 'all_Individuals')
        self.dir_all_poscars = op.join(mother_dir , f'SETUP_COMP{comp}', 'all_poscars')
        self.test_elements = test_elements
        self.flag_one_temp_per_pair = flag_one_temp_per_pair
        self.flag_all_temp = flag_all_temp
        
        self.hyperparameters = hyperparameters

        os.makedirs(self.dir_all_Individuals, exist_ok=True)
        os.makedirs(self.dir_all_poscars, exist_ok=True)

        self.gen_pairs = []
        self.count_ea_searches = 0
        if comp == 1:
            for i in range(len(test_elements)):
                for j in range(i+1, len(test_elements)):
                    elAelB = [test_elements[i], test_elements[j]]
                    elAelB.sort()
                    self.gen_pairs.append(elAelB)
        else:
            for i in test_elements:
                for j in test_elements:
                    if i == j:
                        continue
                    elAelB = [i, j]
                    self.gen_pairs.append(elAelB)

        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/A{comp}B.json'), 'r') as f:
            self.ent_dict = json.load(f)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/EntGS.json'), 'r') as f:
            self.gs_dict = json.load(f)


        self.comp = int(comp)
        self.num_template = 0
        self.n_fails = 0
        

        self.pairs = [] #tuple like (pair, sg)
        self.banned_pairs = [] #pairs with no available templates (Already  chosen or non existent)
        self.poscars = []
        
        self.from_scratch = True
        self.flag_conv = True

        self.trial_SG = None
        self.trial_poscar = None
        self.trial_SG_list = []
        self.trial_poscar_list = []
        
        self.trial_pair = None

        
        if restart_file is not None:
            self.from_scratch = False
            
            flag_pairs=False
            flag_ent=False
            flag_idx=False
            flag_poscars=False
            flag_ea=False
            
            with open(restart_file, 'r') as input:
                lines=input.readlines()
            
            for line in lines:
                if line.startswith('NUMBER OF TEMPLATES'):
                    self.num_template = int(re.search(r'\d+',line).group())
                    self.data = np.zeros((2, self.num_template, self.num_template))
                if line.startswith('COMPOSITION'):
                    self.comp= float(re.search(r'\d+(.\d+)?', line).group())
                if flag_poscars:
                    if line.startswith('EA'):
                        if flag_ea:
                            self.poscars.append(poscar_str)
                        flag_ea = True
                        poscar_str=''
                    poscar_str+=line
                if line.startswith('POSCARS'):
                    flag_poscars = True
                    flag_pairs = False                    
                if flag_pairs:
                    new_tuple = (line[line.find('[')+1:line.find(']')].split(), int(line[line.find(',')+1:]) )
                    self.pairs.append(new_tuple)
                if line.startswith('pairs'):
                    flag_pairs = True      
            self.poscars.append(poscar_str)
            
        return

    def is_not_in_pair(self, SG):
        for i in self.pairs:
            if i[1] % 1000 == SG:
                return False
        return True
    
    def try_new_pair(self):
        count_flag = 0

        while count_flag <= 210:
            count_flag+=1

            self.trial_pair = None
            self.trial_SG = None
            self.trial_poscar = None

            self.trial_poscar_list = []
            self.trial_SG_list = []


            extraction_list = [x for x in self.gen_pairs if x not in self.banned_pairs]
            
            random_element_pair = random.sample(extraction_list,1)[0]
            self.count_ea_searches += 1
            
            A=random_element_pair[0]
            B=random_element_pair[1]
            
            with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                    fstdout.write(f'Trying generating new template with: {A+B} (Try #{count_flag})\n')

            df_individuals = read_individuals(self.dir_all_Individuals+f'/{A+B}_Individuals')
            P, SG, keepit = best_structures(df_individuals, 0.1, self.dir_all_poscars+f'/{A+B}_gatheredPOSCARS', other_sg_min_acceptable = 75)
            SG = [i for idx, i in enumerate(SG) if keepit[idx]]
            P = [i for idx, i in enumerate(P) if keepit[idx]]    
            if len(SG) > 0:
                with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                        fstdout.write(f'There are {len(SG)} possible templates: {SG}\n')
                self.trial_pair = random_element_pair
                sg_of_pair = []
                for k in range(len(SG)):
                    sg_of_pair.append(int(SG[k]))
                    self.trial_poscar_list.append(P[k])
                    self.trial_SG_list.append(int(SG[k]) + 1000 * (sg_of_pair.count(int(SG[k]))-1))
        
                break   
            else:
                with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                        fstdout.write(f'No structures with high simmetry near ground state: choosign new pair\n')
                self.banned_pairs.append(random_element_pair)
                continue
        
        
        with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                fstdout.write(f'The pair {A+B} tries to make a structure with spacegroup {", ".join([str(i) for i in self.trial_SG_list])} as the {"-th, ".join([str(i) for i in range(self.num_template+1, self.num_template + 1 + len(self.trial_SG_list))])} templates\n')
        return


    def update(self):
        new_tuple = (self.trial_pair, self.trial_SG)
        self.pairs.append(new_tuple)
        if self.trial_pair not in self.banned_pairs:
            with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                    fstdout.write(f'The pair {self.trial_pair[0]+self.trial_pair[1]} has been already chosen: the pair won\'t be sorted again\n')
            self.banned_pairs.append(self.trial_pair)

        self.poscars.append(self.trial_poscar)
        self.num_template += 1
        with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
            fstdout.write(f'The new template has been added to the set succesfully: {self.trial_pair[0]+self.trial_pair[1]} {self.trial_SG} \n')
   

    
    def recap_tempset(self, file = None):
        if file is not None:
            relax_file = file
        else:
            relax_file = f'TemplateSet_{self.comp}'
        with open(relax_file, 'w') as file:
            file.write(f'NUMBER OF TEMPLATES {self.num_template}\n')
            file.write(f'COMPOSITION A {self.comp} B\n')
            file.write('pairs , SPACEGROUPS \n')
            for i, pair in enumerate(self.pairs):
                file.write(f'[{pair[0][0]} {pair[0][1]}] , {pair[1]} \n')    
            file.write('POSCARS\n')
            for i in self.poscars:
                file.write(str(i))
        return
    
    def err_before(self, file_critical_pairs = None):
        if file_critical_pairs:
            with open(file_critical_pairs, 'r') as f:
                crit_pairs_dict = json.load(f)

        differences = []
        if self.comp == 1:
            for k in range(len(self.test_elements)):
                for l in range(k+1,len(self.test_elements)):
                    cp = [self.test_elements[k], self.test_elements[l]]
                    cp.sort()
                    try_pair = cp[0]+cp[1]
                    if file_critical_pairs:
                        if try_pair in crit_pairs_dict[f'{int(self.comp)}'].keys():
                            continue

                    ent_gs = self.gs_dict[str(int(self.comp))][try_pair]   
                    ent_temp = np.zeros(self.num_template)
                    for id_template in range(self.num_template):
                        ent_temp[id_template] = self.ent_dict[try_pair][f'{self.pairs[id_template][0][0]}{self.pairs[id_template][0][1]}_{self.pairs[id_template][1]}'] 
                    ent_temp.sort()
                    differences.append( -( ent_gs - ent_temp[0] )) 
            err = 0
            for i in differences:
                err += max(i/len(differences), 0)
        else:
            for k in self.test_elements:
                for l in self.test_elements:
                    if k == l:
                        continue
                    try_pair = k+l
                    if file_critical_pairs:
                        if try_pair in crit_pairs_dict[f'{int(self.comp)}'].keys():
                            continue

                    ent_gs = self.gs_dict[str(int(self.comp))][try_pair]   
                    ent_temp = np.zeros(self.num_template)
                    for id_template in range(self.num_template):
                        ent_temp[id_template] = self.ent_dict[try_pair][f'{self.pairs[id_template][0][0]}{self.pairs[id_template][0][1]}_{self.pairs[id_template][1]}'] 
                    ent_temp.sort()
                    differences.append( -( ent_gs - ent_temp[0] )) 
            err = 0
            for i in differences:
                err += max(i/len(differences), 0)
        return err



class PairSet:
    def __init__(self, template_set, test_elements, hyperparameters, dist_function = levensthein_distance, relaxed_pairs = None, comp=1, mother_dir = './SETUP_FILES/') -> None:
        
        #DA METTERE INDIPENDENZA DA TEMPLATE SET
        self.test_elements=test_elements
        self.dist_function = dist_function 
        self.from_scratch = True   
        self.num_pairs = 0
        self.n_fails = 0
        self.num_template = template_set.num_template
        self.comp = template_set.comp
        self.hyperparameters = hyperparameters

        self.possible_pairs = []
        if comp == 1:
            for i in range(len(test_elements)):
                for j in range(i+1, len(test_elements)):
                    elAelB = [test_elements[i], test_elements[j]]
                    elAelB.sort()
                    self.possible_pairs.append(elAelB)
        else:
            for i in test_elements:
                for j in test_elements:
                    if i == j:
                        continue
                    elAelB = [i, j]
                    self.possible_pairs.append(elAelB)

        self.pairs = []
        self.banned_pairs = []
        self.poscars = template_set.poscars
        self.sg = []
        self.gen_pairs = []
        for i in template_set.pairs:
            self.sg.append(i[1])
            self.gen_pairs.append(i[0])

        self.data = np.zeros((3, self.num_template, self.num_pairs))

        if relaxed_pairs is not None:
            self.from_scratch = False

            flag_pairs=False
            flag_ent=False
            flag_idx_2=False
            flag_idx=False

            with open(relaxed_pairs, 'r') as input_file:
                lines=input_file.readlines()
            
            for line in lines:
                if line.startswith('NUMBER OF PAIRS'):
                    self.num_pairs= int(re.search(r'\d+', line).group())
                if line.startswith('COMPOSITION A'):
                    self.comp= int(re.search(r'\d+', line).group())
                if line.startswith('NUMBER OF TEMPLATES'):
                    self.num_template= int(re.search(r'\d+', line).group())
                    self.data = np.zeros((3, self.num_template, self.num_pairs))
                if flag_idx_2:
                    self.data[2,int(line[:line.find(':')])]=(np.array(line[line.find('[')+1:line.find(']')].split())).astype(float)  
                if line.startswith('R(P) IDX'):
                    flag_idx_2 = True
                    flag_idx = False
                if flag_idx:
                    self.data[1,int(line[:line.find(':')])]=(np.array(line[line.find('[')+1:line.find(']')].split())).astype(float)  
                if line.startswith('R(T) IDX'):
                    flag_idx = True
                    flag_ent = False
                if flag_ent:
                    self.data[0,int(line[:line.find(':')])]=(np.array(line[line.find('[')+1:line.find(']')].split())).astype(float)
                if line.startswith('RANKING VECTORS ENTHALPIES'):
                    flag_ent = True
                    flag_pairs = False                    
                if flag_pairs:
                    self.pairs.append(line[line.find('[')+1:line.find(']')].split())
                if line.startswith('RELAXED pairs'):
                    flag_pairs = True      
        

        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/OneEl.json'), 'r') as f:
            self.one_el_dict = json.load(f)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/A{self.comp}B.json'), 'r') as f:
            self.ent_dict = json.load(f)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/EntGS.json'), 'r') as f:
            self.gs_dict = json.load(f)

        pass


    def add_pair(self):
        tries=0
        while True:
            if tries>=5:
                with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                        fstdout.write('WARNING: too many tries, no new pair added\n')
                return
            tries+=1

            while True:
                extraction_list = [x for x in self.possible_pairs if (x not in self.banned_pairs and x not in self.pairs)]
                random_element_pair = random.sample(extraction_list,1)[0]
                if self.comp == 1:
                    random_element_pair.sort()
                if random_element_pair in self.banned_pairs:
                    continue
                elif random_element_pair not in self.pairs:
                    A=random_element_pair[0]
                    B=random_element_pair[1]
                break


            new_ranking = np.zeros(self.num_template)
            for i in range(self.num_template):

            
                convergence_flag = True # mqe.check_convergence(self.dir_all_qeoutput + f'{i}.out')
                
                if not convergence_flag:

                    self.n_fails+=1
                    self.banned_pairs.append([A,B])
                    break
                ent_form =  ( self.one_el_dict[A] * self.comp + self.one_el_dict[B] ) / (self.comp + 1 )
                new_ranking[i] = self.ent_dict[A+B][f'{self.gen_pairs[i][0]+self.gen_pairs[i][1]}_{self.sg[i]}'] - ent_form 
            if not convergence_flag:

                return
            else:
                break

        self.data = np.pad(self.data, ((0, 0), (0, 0), (0, 1)), constant_values=0)
        self.data[0,:,-1] = new_ranking
        self.data[1,:,-1] = self.num_pairs
        self.data[2,:,-1] = np.arange(0, self.num_template, 1)
        self.pairs.append([A,B])
        self.num_pairs += 1
        return
    
    def recap_relaxed(self, outfile = 'RelaxedPairs.txt'):
        with open(outfile, 'w') as file:
            file.write(f'NUMBER OF PAIRS {self.num_pairs}\n')
            file.write(f'COMPOSITION A {self.comp} B\n')
            file.write(f'NUMBER OF TEMPLATES {self.num_template}\n')
            file.write('RELAXED pairs\n')
            for i, pair in enumerate(self.pairs):
                file.write(f'[{pair[0]} {pair[1]}] \n')  
            file.write('RANKING VECTORS ENTHALPIES\n')
            for i in range(self.num_template):
                arr = str(self.data[0,i]).replace('\n','')
                file.write(f'{i}:{arr} \n')
            file.write('R(T) IDX\n')
            for i in range(self.num_template):
                arr = str(self.data[1,i]).replace('\n','')
                file.write(f'{i}:{arr} \n')
            file.write('R(P) IDX\n')
            for i in range(self.num_template):
                arr = str(self.data[2,i]).replace('\n','')
                file.write(f'{i}:{arr} \n')
        return

    def order_wrt_templates(self): #moves the templates
        # Oreder the matrix by columns of first matrix
        matrix = self.data.copy()
        matrix = matrix.transpose((0,2,1))
        sorted_indices = np.argsort(matrix[0], axis=1)
        for i in range(matrix.shape[1]):
            matrix[0,i] = matrix[0,i, sorted_indices[i]]
            matrix[1,i] = matrix[1,i, sorted_indices[i]]
            matrix[2,i] = matrix[2,i, sorted_indices[i]]
        matrix = matrix.transpose((0,2,1))
        return matrix

    def dist_matrix(self):
        # Compute the distance matrix between the templates moving the pairs
        dist = np.zeros((self.num_template, self.num_template))
        matrix = self.data.copy()
        for i in range(self.num_template):
            for j in range(self.num_template):
                dist[i,j] = self.dist_function(matrix[:,i,:], matrix[:,j,:])
        
        return dist
    

    def template_gs(self):
        # Compute how many times each template is the ground state
        ist = np.zeros(self.num_template)
        matrix = self.order_wrt_templates()
        for i in range(self.num_pairs):
            idx = int(matrix[2,0,i])
            ist [idx] += 1
        return ist
    
    def formation_percentage(self):
        # Compute the fraction of negative formation enthalpy for each template
        form_negative = np.zeros(self.num_template)
        for i in range(self.num_template):
            form_negative[i] = np.sum(self.data[0,i] < 0)/self.num_pairs
        return form_negative

    def reduced_set(self):
        if 'n_final_templates' not in self.hyperparameters.keys():
            # Compute the reduced set of templates
            form_negative = self.formation_percentage()
            ist = self.template_gs()
            sg = self.sg.copy()

            set_of_templates = [x for x in range(self.num_template)]
            lev_matrix = self.dist_matrix()
            np.fill_diagonal(lev_matrix, 10)

            while lev_matrix.min() < self.hyperparameters['lev_red']:
                idx = np.unravel_index(np.argmin(lev_matrix), lev_matrix.shape)
                i = idx[0]
                j = idx[1]

                a_j = form_negative[j] * self.hyperparameters['weight_formation_entalphy'] 
                b_j = ist[j] * self.hyperparameters['weight_occurrence']/self.num_pairs
                c_j = sg[j] * self.hyperparameters['weight_sg']

                a_i = form_negative[i] * self.hyperparameters['weight_formation_entalphy']
                b_i = ist[i] * self.hyperparameters['weight_occurrence']/self.num_pairs
                c_i = sg[i] * self.hyperparameters['weight_sg']

                score_j = a_j + b_j + c_j
                score_i = a_i + b_i + c_i
                
                if score_j > score_i:
                    set_of_templates = np.delete(set_of_templates, i)
                    lev_matrix = np.delete(lev_matrix, i, axis=0)
                    lev_matrix = np.delete(lev_matrix, i, axis=1)
                    form_negative = np.delete(form_negative, i)
                    ist = np.delete(ist, i)
                    sg = np.delete(sg, i)
                else:
                    set_of_templates = np.delete(set_of_templates, j)
                    lev_matrix = np.delete(lev_matrix, j, axis=0)
                    lev_matrix = np.delete(lev_matrix, j, axis=1)
                    form_negative = np.delete(form_negative, j)
                    ist = np.delete(ist, j)
                    sg = np.delete(sg, j)


        else:
            # Compute the reduced set of templates
            form_negative = self.formation_percentage()
            ist = self.template_gs()
            sg = self.sg.copy()

            set_of_templates = [x for x in range(self.num_template)]
            lev_matrix = self.dist_matrix()
            np.fill_diagonal(lev_matrix, 10)

            while len(set_of_templates) > self.hyperparameters['n_final_templates']:
                idx = np.unravel_index(np.argmin(lev_matrix), lev_matrix.shape)
                i = idx[0]
                j = idx[1]

                a_j = form_negative[j] * self.hyperparameters['weight_formation_entalphy'] 
                b_j = ist[j] * self.hyperparameters['weight_occurrence']/self.num_pairs
                c_j = sg[j] * self.hyperparameters['weight_sg']

                a_i = form_negative[i] * self.hyperparameters['weight_formation_entalphy']
                b_i = ist[i] * self.hyperparameters['weight_occurrence']/self.num_pairs
                c_i = sg[i] * self.hyperparameters['weight_sg']

                score_j = a_j + b_j + c_j
                score_i = a_i + b_i + c_i
                        
                if score_j > score_i:
                    set_of_templates = np.delete(set_of_templates, i)
                    lev_matrix = np.delete(lev_matrix, i, axis=0)
                    lev_matrix = np.delete(lev_matrix, i, axis=1)
                    form_negative = np.delete(form_negative, i)
                    ist = np.delete(ist, i)
                    sg = np.delete(sg, i)
                else:
                    set_of_templates = np.delete(set_of_templates, j)
                    lev_matrix = np.delete(lev_matrix, j, axis=0)
                    lev_matrix = np.delete(lev_matrix, j, axis=1)
                    form_negative = np.delete(form_negative, j)
                    ist = np.delete(ist, j)
                    sg = np.delete(sg, j)
                

        return set_of_templates

    def error_single_composition(self):
        # Compute the error of the single composition
        differences = []
        set_of_remaining_templates = self.reduced_set()

        if self.comp == 1:
            for k in range(len(self.test_elements)):
                for l in range(k+1,len(self.test_elements)):
                
                    cp = [self.test_elements[k], self.test_elements[l]]
                    cp.sort()
                    try_pair = cp[0]+cp[1]

                    ent_gs = self.gs_dict[str(int(self.comp))][try_pair]  
                    ent_temp = np.zeros(len(set_of_remaining_templates))
                    for j, i in enumerate(set_of_remaining_templates):
                        ent_temp[j] = self.ent_dict[try_pair][f'{self.gen_pairs[i][0]}{self.gen_pairs[i][1]}_{self.sg[i]}'] 
                    ent_temp.sort()
                    differences.append( -( ent_gs - ent_temp[0] )) 

        else:
            for k in range(len(self.test_elements)):
                for l in range(len(self.test_elements)):
                    if k == l:
                        continue
                    cp = [self.test_elements[k], self.test_elements[l]]
                    try_pair = cp[0]+cp[1]

                    ent_gs = self.gs_dict[str(int(self.comp))][try_pair]  
                    ent_temp = np.zeros(len(set_of_remaining_templates))
                    for j, i in enumerate(set_of_remaining_templates):
                        ent_temp[j] = self.ent_dict[try_pair][f'{self.gen_pairs[i][0]}{self.gen_pairs[i][1]}_{self.sg[i]}'] 
                    ent_temp.sort()
                    differences.append( -( ent_gs - ent_temp[0] )) 
        return differences

    def total_error(self):
        # Compute the total error of the reduced set
        differences = self.error_single_composition()
        err = 0
        for i in differences:
            err += max(i/len(differences), 0)
        return err
    

def generate_one_templateset(hyperparameters, test_elements, dist_function = levensthein_distance, flag_one_temp_per_pair = False):
    template = TemplateSet(test_elements=test_elements, hyperparameters=hyperparameters, comp = hyperparameters['comp'], flag_one_temp_per_pair = flag_one_temp_per_pair)

    if template.from_scratch:
        template.try_new_pair()
        for trial_poscar, trial_sg in zip(template.trial_poscar_list, template.trial_SG_list):
            template.trial_poscar = trial_poscar
            template.trial_SG = trial_sg
            template.update()

    while template.num_template < hyperparameters['n_template']:

        # Try a new pair
        template.try_new_pair()
        for trial_poscar, trial_sg in zip(template.trial_poscar_list, template.trial_SG_list):
            template.trial_poscar = trial_poscar
            template.trial_SG = trial_sg
            template.update(hyperparameters)

    return template

def generate_one_pairset (template_prod, hyperparameters, test_elements, dist_function = levensthein_distance):
    reduction_set = PairSet(template_prod, test_elements, hyperparameters, dist_function, comp=hyperparameters['comp'])
    npair = 0
    while reduction_set.num_pairs < hyperparameters['n_pairs']:
        npair+=1 
        reduction_set.add_pair()
    return reduction_set


def graph_difference_std (dif_mean, dif_std, n_temp, c_value, dir_temp, hyperparameters, test_elements):
    fig, ax1 = plt.subplots(1,1, figsize=(25, 10))
    differences = dif_mean
    color_value = cm.viridis(c_value)

    ax1.bar(np.arange(0,len(differences),1), differences, yerr = dif_std , color=color_value, edgecolor='black', alpha=0.8)
    ax1.set_title(r'$\Delta H$ for each pair with '+f'{n_temp} templates')
    ax1.set_xticks(np.arange(0,len(differences),1))
    if hyperparameters['comp'] == 1:
        temp_ticks = []
        for i in range(len(test_elements)):
            for j in range(i+1,len(test_elements)):
                pair = [test_elements[i], test_elements[j]]
                pair.sort()
                temp_ticks.append(pair[0]+pair[1])

        ax1.set_xticklabels(temp_ticks, rotation=90)
    else:
        ax1.set_xticklabels([f'{test_elements[k]}{test_elements[l]}' for k in range(len(test_elements)) for l in range(len(test_elements)) if k!=l], rotation=90)
    ax1.grid(linestyle=':')
    ax1.set_ylabel('Enthalpy difference (eV/atom)')
    ax1.set_ylim(-1, 2)
    cartella = dir_temp+f'{n_temp}/'
    nome_file = f'Differeces_{n_temp}.png'
    percorso_completo = os.path.join(cartella, nome_file)
    fig.savefig(percorso_completo)
    
    return
