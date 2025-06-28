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


class InitialSet:
    """
        The class used to generate the initial set of templates and to evaluate its performance.
        Args:
            test_elements (list): The list of elements to be used.
            hyperparameters (dict): The hyperparameters of the template set.
            restart_file (str): The path of the file to be used to restart the template set. If None, it will be created from scratch.
            mother_dir (str): The path of the directory where the USPEX OUT files are stored and where file related to the template set will be stored.
        """
    def __init__(self, test_elements, hyperparameters, restart_file = None, mother_dir = './TEMPLATES/') -> None:
        
        self.test_elements = test_elements
        
        self.hyperparameters = hyperparameters

        self.possible_pairs = []
        self.count_ea_searches = 0

        self.comp = int(hyperparameters['comp'])

        self.dir_all_Individuals = op.join(mother_dir , f'SETUP_COMP{self.comp}', 'all_Individuals')
        self.dir_all_poscars = op.join(mother_dir , f'SETUP_COMP{self.comp}', 'all_poscars')

        if self.comp == 1:
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

        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/A{self.comp}B.json'), 'r') as f:
            self.ent_dict = json.load(f)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/EntGS.json'), 'r') as f:
            self.gs_dict = json.load(f)

        self.num_template = 0

        # self.n_fails = 0
        # self.flag_conv = True
        
        self.generative_pairs = []          # tuple like (pair, sg)
        self.banned_pairs = []  
        if self.comp == 6:
            self.banned_pairs = [['C','B']]            # pairs with no available templates (Already  chosen or non existent)
        # self.poscars = []
        
        self.from_scratch = True

        self.trial_pair = None              # just list of the elements in the pair
        self.trial_SG = None
        # self.trial_poscar = None
        self.trial_SG_list = []
        # self.trial_poscar_list = []
        
        
        if restart_file is not None:        # we are loading a previous initial set
            self.from_scratch = False
            
            flag_pairs=False
            # flag_poscars=False
            # flag_first_poscar = False
            
            with open(restart_file, 'r') as input:
                lines=input.readlines()
            
            for line in lines:
                if line.startswith('NUMBER OF TEMPLATES'):
                    self.num_template = int(re.search(r'\d+',line).group())
                    self.data = np.zeros((2, self.num_template, self.num_template))
                if line.startswith('COMPOSITION'):
                    self.comp= float(re.search(r'\d+(.\d+)?', line).group())

                # if flag_poscars:
                #     if line.startswith('EA'):
                #         if flag_first_poscar:
                #             self.poscars.append(poscar_str)
                #         flag_first_poscar = True
                #         poscar_str=''
                #     poscar_str+=line
                # if line.startswith('POSCARS'):
                #     flag_poscars = True
                #     flag_pairs = False        
                            
                if flag_pairs:
                    new_tuple = (line[line.find('[')+1:line.find(']')].split(), int(line[line.find(',')+1:]) )
                    self.generative_pairs.append(new_tuple)
                if line.startswith('PAIRS'):
                    flag_pairs = True      

            # self.poscars.append(poscar_str)
        return
    
    def try_new_pair(self):

        """
        Tries to generate a new template by using the EA search.
        It sorts a random pair of elements from the possible pairs and launches the EA search.
        Then it checks if there are any templates with high symmetry near the ground state.
        If no templates are found, it bans the pair and tries again.
        Else, it adds the new templates to the set.
        """


        count_flag = 0

        while count_flag <= 210:

            count_flag+=1

            self.trial_pair = None
            self.trial_SG = None
            # self.trial_poscar = None

            self.trial_SG_list = []
            # self.trial_poscar_list = []

            extraction_list = [x for x in self.possible_pairs if x not in self.banned_pairs]
            random_element_pair = random.sample(extraction_list,1)[0]

            self.count_ea_searches += 1
            
            A=random_element_pair[0]
            B=random_element_pair[1]
            
            with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                    fstdout.write(f'Trying generating new template with: {A+B} (Try #{count_flag})\n')          # Here the EA search is launched

            df_individuals = read_individuals(self.dir_all_Individuals+f'/{A+B}_Individuals')                   # EA search output
            _ , SG = best_structures(df_individuals, 0.1, self.dir_all_poscars+f'/{A+B}_gatheredPOSCARS')

            if len(SG) > 0:
                with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                    fstdout.write(f'There are {len(SG)} possible templates with SG: {", ".join([str(i) for i in SG])}\n')

                self.trial_pair = random_element_pair
                sg_of_pair = []
                for k in range(len(SG)):
                    sg_of_pair.append(int(SG[k]))
                    # self.trial_poscar_list.append(P[k])
                    self.trial_SG_list.append(int(SG[k]) + 1000 * (sg_of_pair.count(int(SG[k]))-1))
        
                break   
            else:
                with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                        fstdout.write(f'No structures with high symmetry near ground state: choosing new pair\n')
                self.banned_pairs.append(random_element_pair)
                continue
        
        
        with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                fstdout.write(f'The pair {A+B} tries to make structures with spacegroup {", ".join([str(i) for i in self.trial_SG_list])} as the {" and ".join([str(i) for i in range(self.num_template + 1, self.num_template + 1 + len(self.trial_SG_list))])} templates\n')
        
        return


    def update(self):

        """
        Updates the template set with the new templates generated by the EA search.
        It adds the new templates to the set and bans the generative pair used.
        """

        new_tuple = (self.trial_pair, self.trial_SG)
        self.generative_pairs.append(new_tuple)

        if self.trial_pair not in self.banned_pairs:
            with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
                    fstdout.write(f'The pair {self.trial_pair[0]+self.trial_pair[1]} will not be sorted again\n')
            self.banned_pairs.append(self.trial_pair)

        # self.poscars.append(self.trial_poscar)
        self.num_template += 1
        with open(f'log{self.hyperparameters["job_id"]}.txt','a') as fstdout:
            fstdout.write(f'The new template has been added to the set succesfully: {self.trial_pair[0]+self.trial_pair[1]} {self.trial_SG} \n')
   

    
    def recap(self, file_path = 'InitialSet.txt'):
        """
        Recaps the template set in a file.
        It writes the number of templates, the composition, the pairs and the spacegroups.

        Args:
            file_path (str): The path of the file where to write the template set. If default it writes in the default path: 'InitialSet.txt'.
        """

        with open(file_path, 'w') as f:
            f.write(f'NUMBER OF TEMPLATES {self.num_template}\n')
            f.write(f'COMPOSITION A {self.comp} B\n')
            f.write('PAIRS , SPACEGROUPS \n')
            for i, pair in enumerate(self.generative_pairs):
                f.write(f'[{pair[0][0]} {pair[0][1]}] , {pair[1]} \n')    
            # f.write('POSCARS\n')
            # for i in self.poscars:
            #     f.write(str(i))
        return
    
    def difference_from_uspex(self, file_critical_pairs = None):

        """
        Computes the accuracy of the template set before the reduction.
        It compares the enthalpy predicted by USPEX with the enthalpy predicted by the template set considering it 0 if template set is better.
        Averages the difference over all possible pairs of elements in the test set.
        If the file_critical_pairs is provided, it excludes the pairs in the file from the error calculation.
        Args:
            file_critical_pairs (str): The path of the file with the critical pairs. If None, they are not excluded.
        Returns:
            av_diff (float): The average difference between the enthalpy predicted by USPEX and the enthalpy predicted by the template set.
        """

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
                        ent_temp[id_template] = self.ent_dict[try_pair][f'{self.generative_pairs[id_template][0][0]}{self.generative_pairs[id_template][0][1]}_{self.generative_pairs[id_template][1]}'] 

                    differences.append( -( ent_gs - min(ent_temp) )) 
            av_diff = 0
            for i in differences:
                av_diff += max(i/len(differences), 0)
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
                        ent_temp[id_template] = self.ent_dict[try_pair][f'{self.generative_pairs[id_template][0][0]}{self.generative_pairs[id_template][0][1]}_{self.generative_pairs[id_template][1]}']

                    differences.append( -( ent_gs - min(ent_temp) )) 
            av_diff = 0
            for i in differences:
                av_diff += max(i/len(differences), 0)
        return av_diff



class FinalSet:
    def __init__(self, template_set, test_elements, hyperparameters, restart_file = None) -> None:
        
        #DA METTERE INDIPENDENZA DA TEMPLATE SET
        self.test_elements = test_elements
        self.dist_function = levensthein_distance

        self.from_scratch = True   
        
        self.num_pairs = 0
        # self.n_fails = 0
        
        self.num_init_template = template_set.num_template
        self.comp = int(hyperparameters['comp'])
        self.hyperparameters = hyperparameters

        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/OneEl.json'), 'r') as f:
            self.one_el_dict = json.load(f)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/A{self.comp}B.json'), 'r') as f:
            self.ent_dict = json.load(f)
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'ENTHALPY/EntGS.json'), 'r') as f:
            self.gs_dict = json.load(f)


        self.possible_pairs = []
        if self.comp == 1:
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

        self.count_crit_in_validation = 0

        if 'critical_pairs' in self.hyperparameters.keys():                     # if the critical pairs are provided, we load them and we exclude them from the possible validation pairs
            with open(self.hyperparameters['critical_pairs'], 'r') as f:
                crit_pairs_dict = json.load(f)
            if str(self.comp) in crit_pairs_dict.keys():
                for i in crit_pairs_dict[str(self.comp)].keys():
                    self.banned_pairs.append([i[0], i[1]])

        # Load the INITIAL SET
        # self.poscars = template_set.poscars
        self.sg = []
        self.init_temp_names = []

        for i in template_set.generative_pairs:
            self.sg.append(i[1])
            self.init_temp_names.append(f'{i[0][0]}{i[0][1]}_{i[1]}')

        self.num_final_template = 0
        self.final_temp_names = None
        
        self.data = np.zeros((3, self.num_init_template, self.num_pairs))  # matrix of the enthalpies of relaxed structures obtained by ralaxing validation pairs on templates in initial set
        
        if restart_file is not None:
            self.from_scratch = False

            flag_pairs=False
            flag_final_temp=False

            with open(restart_file, 'r') as input_file:
                lines=input_file.readlines()
            
            for line in lines:
                if line.startswith('NUMBER OF PAIRS'):
                    self.num_pairs = int(re.search(r'\d+', line).group())
                
                if line.startswith('COMPOSITION A'):
                    self.comp = int(re.search(r'\d+', line).group())
                
                if line.startswith('NUMBER OF TEMPLATES'):
                    self.num_final_template = int(re.search(r'\d+', line).group())
                    self.data = np.zeros((3, self.num_final_template, self.num_pairs))

                if flag_final_temp:
                    self.final_temp_names.append(line.strip())
                if line.startswith('TEMPLATE NAMES'):
                    flag_final_temp = True
                    self.final_temp_names = []
                    flag_pairs = False                    
                
                if flag_pairs:
                    self.pairs.append(line[line.find('[')+1:line.find(']')].split())
                if line.startswith('USED PAIRS'):
                    flag_pairs = True   

            # CREATE RANKING VECTORS
            for i in range(self.num_final_template):
                for j in range(self.num_pairs):
                    self.data[0,i,j] = self.ent_dict[self.pairs[j][0]+self.pairs[j][1]][f'{self.final_temp_names[i]}'] - (self.one_el_dict[self.pairs[j][0]] * self.comp + self.one_el_dict[self.pairs[j][1]]) / (self.comp + 1)
                    self.data[1,i,j] = j

        return


    def add_pair(self, file_crit_pairs = None):

        """
        Adds a new pair to the template set.
        New pair is sorted from all possible pairs with test elements.
        It generates a new column in the data matrix with the enthalpy of the new pair for each template in the initial set, 
        in order to add the new element to each template ranking vector.
        If the file_critical_pairs is provided, the number of critical pairs in the validation set is counted.
        
        Args:
            file_crit_pairs (str): The path of the file with the critical pairs. If None, they are not counted.
        """
        
        extraction_list = [x for x in self.possible_pairs if x not in self.banned_pairs]
        random_element_pair = random.sample(extraction_list,1)[0]

        A = random_element_pair[0]
        B = random_element_pair[1]

        new_elements_of_ranking_vector = np.zeros(self.num_init_template)               # they are the elements of the ranking vector related to the new pair for each template in the initial set: a new column where template ranking vectors are rows    
        for i in range(self.num_init_template):

            ent_form =  ( self.one_el_dict[A] * self.comp + self.one_el_dict[B] ) / (self.comp + 1 )
            new_elements_of_ranking_vector[i] = self.ent_dict[A+B][f'{self.init_temp_names[i]}'] - ent_form 

    
        self.data = np.pad(self.data, ((0, 0), (0, 0), (0, 1)), constant_values=0)
        self.data[0,:,-1] = new_elements_of_ranking_vector - new_elements_of_ranking_vector.min()
        self.data[1,:,-1] = self.num_pairs                                          # Id of the new pair
        self.pairs.append([A,B])
        self.banned_pairs.append([A,B])

        if file_crit_pairs:
            with open(file_crit_pairs, 'r') as f:
                crit_pairs = json.load(f)
            
            if A+B in crit_pairs[f'{int(self.comp)}'].keys():
                self.count_crit_in_validation += 1

        self.num_pairs += 1
        return
    
    def recap(self, file_path = 'FinalSet.txt'):
        """
        Recaps the final set in a file.
        It writes the number of pairs, the composition, the number of templates and their names.
        It also writes the ranking vectors and the enthalpies of the templates in the initial set.
        Args:
            file_path (str): The path of the file where to write the template set. If default it writes in the default path: 'FinalSet.txt'.
        """

        with open(file_path, 'w') as f:
            f.write(f'NUMBER OF PAIRS {self.num_pairs}\n')
            f.write(f'COMPOSITION A {self.comp} B\n')
            f.write(f'NUMBER OF TEMPLATES {self.num_final_template}\n')
            f.write('USED PAIRS\n')
            for i, pair in enumerate(self.pairs):
                f.write(f'[{pair[0]} {pair[1]}] \n')  
            f.write('TEMPLATE NAMES\n')
            for i in range(self.num_final_template):
                f.write(f'{self.final_temp_names[i]} \n')
        return

    def dist_matrix(self):
        
        '''
        Compute the distance matrix between the templates ranking vectors.

        Returns:
            dist (np.ndarray): The distance matrix between the templates ranking vectors. (num_init_template x num_init_template)
        '''
        
        dist = np.zeros((self.num_init_template, self.num_init_template))
        matrix = self.data.copy()
        for i in range(self.num_init_template):
            for j in range(self.num_init_template):
                dist[i,j] = self.dist_function(matrix[:,i,:], matrix[:,j,:])
        
        return dist
    
    def template_gs(self):
        '''
        Compute the number of times each template is the lowest energy template for each pair. Normalized by the number of pairs.

        Returns:
            ist (np.ndarray): The number of times each template is the lowest energy template for each pair. (num_init_template)
        '''

        ist = np.zeros(self.num_init_template)
        matrix = self.data.copy()

        for i in range(self.num_pairs):
            idx_template = np.argmin(matrix[0,:,i])
            ist[idx_template] += 1/self.num_pairs
        return ist
    
    def formation_percentage(self):
        '''
        Compute the fraction of negative formation enthalpy for each template. Normalized by the number of pairs.

        Returns:
            form_negative (np.ndarray): The fraction of negative formation enthalpy for each template. (num_init_template)
        '''

        form_negative = np.zeros(self.num_init_template)
        for i in range(self.num_init_template):
            form_negative[i] = np.sum(self.data[0,i] < 0)/self.num_pairs
        return form_negative

    def reduced_set(self):
        '''
        Compute the reduced set of templates. The reduced set is obtained by removing the templates that are too similar to each other.
        The similarity is computed using the distance matrix.
        A score is assigned to each template based on the formation enthalpy, the occurrence and the spacegroup, weighted by the hyperparameters.
        Starting from the two closest templates, the one with the lowest score is removed.

        If n_final_templates is not provided, the reduced set is obtained by removing the templates until the distance matrix is greater than a threshold lev_red.
        If n_final_templates is provided, the reduced set is obtained by removing the templates until the number of templates is equal to n_final_templates.
        The reduced set is returned as a list of template names.

        Returns:
            set_of_templates (list): The reduced set of templates. (num_final_templates)
        '''

        if 'n_final_templates' not in self.hyperparameters.keys():
            form_negative = self.formation_percentage()
            ist = self.template_gs()
            sg = self.sg.copy()

            set_of_templates = [x for x in self.init_temp_names]

            lev_matrix = self.dist_matrix()
            np.fill_diagonal(lev_matrix, np.inf)

            while lev_matrix.min() < self.hyperparameters['lev_red']:
                idx = np.unravel_index(np.argmin(lev_matrix), lev_matrix.shape)
                i = idx[0]
                j = idx[1]

                a_j = form_negative[j] * self.hyperparameters['weight_formation_entalphy'] 
                b_j = ist[j] * self.hyperparameters['weight_occurrence']
                c_j = sg[j] * self.hyperparameters['weight_sg']

                a_i = form_negative[i] * self.hyperparameters['weight_formation_entalphy']
                b_i = ist[i] * self.hyperparameters['weight_occurrence']
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
            form_negative = self.formation_percentage()
            ist = self.template_gs()
            sg = self.sg.copy()

            set_of_templates = [x for x in self.init_temp_names]

            lev_matrix = self.dist_matrix()
            np.fill_diagonal(lev_matrix, np.inf)

            while len(set_of_templates) > self.hyperparameters['n_final_templates']:
                idx = np.unravel_index(np.argmin(lev_matrix), lev_matrix.shape)
                i = idx[0]
                j = idx[1]

                a_j = form_negative[j] * self.hyperparameters['weight_formation_entalphy'] 
                b_j = ist[j] * self.hyperparameters['weight_occurrence']
                c_j = sg[j] * self.hyperparameters['weight_sg']

                a_i = form_negative[i] * self.hyperparameters['weight_formation_entalphy']
                b_i = ist[i] * self.hyperparameters['weight_occurrence']
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
                
        self.final_temp_names = set_of_templates.copy()
        self.num_final_template = len(set_of_templates)

        return set_of_templates

    def difference_from_uspex(self, path_critical_pairs = None):
        """
        Computes the accuracy of the template set after the reduction.
        It compares the enthalpy predicted by USPEX with the enthalpy predicted by the template set considering it 0 if template set is better.
        Averages the difference over all possible pairs of elements in the test set.
        If the path_critical_pairs is provided, it excludes the pairs in the file from the error calculation.
        Args:
            path_critical_pairs (str): The path of the file with the critical pairs. If None, they are not excluded.
        Returns:
            av_diff (float): The average difference between the enthalpy predicted by USPEX and the enthalpy predicted by the template set.
        """

        differences = []

        if self.final_temp_names is None:                                   # if the template is already reduced, we use the reduced set
            set_of_remaining_templates = self.reduced_set()
        else:
            set_of_remaining_templates = self.final_temp_names

        if path_critical_pairs:
            with open(path_critical_pairs, 'r') as f:
                dict_critical_pairs = json.load(f)

        if self.comp == 1:
            for k in range(len(self.test_elements)):
                for l in range(k+1,len(self.test_elements)):
                
                    cp = [self.test_elements[k], self.test_elements[l]]
                    cp.sort()
                    try_pair = cp[0]+cp[1]

                    if path_critical_pairs:
                        if try_pair in dict_critical_pairs[f'{int(self.comp)}'].keys():
                            continue

                    ent_gs = self.gs_dict[str(int(self.comp))][try_pair]  
                    ent_temp = np.zeros(len(set_of_remaining_templates))
                    for j, temp_name in enumerate(set_of_remaining_templates):
                        ent_temp[j] = self.ent_dict[try_pair][f'{temp_name}'] 
                
                    differences.append( -( ent_gs - min(ent_temp) )) 

        else:
            for k in range(len(self.test_elements)):
                for l in range(len(self.test_elements)):
                    if k == l:
                        continue
                    cp = [self.test_elements[k], self.test_elements[l]]
                    try_pair = cp[0]+cp[1]

                    if path_critical_pairs:
                        if try_pair in dict_critical_pairs[f'{int(self.comp)}'].keys():
                            continue

                    ent_gs = self.gs_dict[str(int(self.comp))][try_pair]  
                    ent_temp = np.zeros(len(set_of_remaining_templates))
                    for j, temp_name in enumerate(set_of_remaining_templates):
                        ent_temp[j] = self.ent_dict[try_pair][f'{temp_name}'] 
                    ent_temp.sort()
                    differences.append( -( ent_gs - min(ent_temp) )) 

        av_diff = 0
        for i in differences:
            av_diff += max(i/len(differences), 0)
        return av_diff


def generate_initial_set(hyperparameters, test_elements):
    """
    Generates the initial set of templates by using the InitialSet class.

    Args:
        hyperparameters (dict): The hyperparameters of the template set.
        test_elements (list): The list of elements to be used.
    Returns:
        set1 (InitialSet): The initial set of templates.
    """
    
    set1 = InitialSet(test_elements=test_elements, hyperparameters=hyperparameters)

    while set1.num_template < hyperparameters['n_template']:

        set1.try_new_pair()
        # for trial_poscar, trial_sg in zip(set1.trial_poscar_list, set1.trial_SG_list):
        for trial_sg in set1.trial_SG_list:
            # set1.trial_poscar = trial_poscar
            set1.trial_SG = trial_sg
            set1.update()

    return set1


def generate_final_set(init_set, hyperparameters, test_elements, file_crit_pairs = None):
    """
    Generates the final set of templates by using the FinalSet class.

    Args:
        init_set (InitialSet): The initial set of templates.
        hyperparameters (dict): The hyperparameters of the template set.
        test_elements (list): The list of elements to be used.
        file_crit_pairs (str): The path of the file with the critical pairs. If None, they are not excluded.
    Returns:
        reduction_set (FinalSet): The final set of templates.
    """
    
    reduction_set = FinalSet(init_set, test_elements, hyperparameters)
    npair = 0
    while reduction_set.num_pairs < hyperparameters['n_pairs']:
        npair += 1 
        reduction_set.add_pair(file_crit_pairs)
    return reduction_set


