import numpy as np
import os 
import re
import random
import pandas as pd


from pymatgen.core import Element

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from template_csp.manageuspex import read_individuals, best_structures, copy_files
from template_csp import manageqe as mqe
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

# Compute the Levenshtein distance between two arrays
def levensthein_distance(a1,a2):
    dist=0
    for i in range(len(a1)):
        if a1[i] != a2[i]:
            dist+=1-float(i)/len(a1)
    return dist

class TemplateSet:
    def __init__(self, test_elements, restart_file = None,   comp = 1, mother_dir = './SETUP_FILES/', clusters = None) -> None:
        self.dir_all_Individuals = mother_dir + 'all_Individuals/'
        self.dir_all_poscars = mother_dir + 'all_poscars/'
        self.dir_all_qeoutput = mother_dir + 'all_qeoutput/'
        self.dir_all_qeinput = mother_dir + 'all_qeinput/'
        self.dir_all_cif = mother_dir + 'all_cif/'


        #create_directory(self.dir_all_qeoutput)
        #create_directory(self.dir_all_qeinput)
        #create_directory(self.dir_all_cif)
        create_directory(self.dir_all_Individuals)
        create_directory(self.dir_all_poscars)


        self.gen_couples = []
        if comp == 1:
            for i in range(len(test_elements)):
                for j in range(i+1, len(test_elements)):
                    elAelB = [test_elements[i], test_elements[j]]
                    elAelB.sort()
                    self.gen_couples.append(elAelB)
        else:
            for i in test_elements:
                for j in test_elements:
                    elAelB = [i, j]
                    self.gen_couples.append(elAelB)

        self.df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'A{comp}B/relaxation/RELAX_DATA'), sep=",", index_col=0, na_filter = False)

        self.comp = comp
        self.num_template = 0
        self.n_fails = 0
        
        self.flag_cluster = False
        if clusters is not None:
            self.trial_cluster = None
            self.flag_cluster = True
            self.couples_in_clusters = []
            self.original_clusters = []
            self.freq_cluster = []
            for cluster in clusters.keys():
                self.couples_in_clusters.append([])
                self.original_clusters.append((int(cluster)-1,clusters[cluster]['couples']))
                self.freq_cluster.append(clusters[cluster]['freq'])
            
        self.couples = [] #Couples chosen for the templates
        self.banned_couples = [] #Couples with no available templates (Already  chosen or non existent)
        self.poscars = []
        
        self.from_scratch = True
        self.flag_conv = True

        self.trial_SG = None
        self.trial_poscar = None
        self.trial_couple = None
        
        if restart_file is not None:
            self.from_scratch = False
            
            flag_couples=False
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
                    flag_idx = False
                if flag_idx:
                    self.data[1,int(line[:line.find(':')])]=(np.array(line[line.find('[')+1:line.find(']')].split())).astype(float)
                if line.startswith('RANKING VECTORS IDX'):
                    flag_idx = True
                    flag_ent = False
                if flag_ent:
                    self.data[0,int(line[:line.find(':')])]=(np.array(line[line.find('[')+1:line.find(']')].split())).astype(float)
                if line.startswith('RANKING VECTORS ENTHALPIES'):
                    flag_ent = True
                    flag_couples = False                    
                if flag_couples:
                    new_tuple = (line[line.find('[')+1:line.find(']')].split(), int(line[line.find(',')+1:]) )
                    self.couples.append(new_tuple)
                if line.startswith('COUPLES'):
                    flag_couples = True      
            self.poscars.append(poscar_str)
            
            #for i, poscar in enumerate(self.poscars):
            #    c2q.poscar_to_input(poscar, self.dir_all_cif + f'{i}.cif', self.dir_all_qeinput + f'{i}.in')
        return

    def is_not_in_couple(self, SG):
        for i in self.couples:
            if i[1] == SG:
                return False
        return True
    
    def try_new_couple(self):
        count_flag = 0


        while count_flag<=50:
            count_flag+=1

            self.trial_couple = None
            self.trial_poscar = None
            self.trial_SG = None

            if self.flag_cluster:
                self.trial_cluster = None
                flag_cluster_found = False

                idx_empty_clusters = []
                for idx, cluster in enumerate(self.couples_in_clusters):
                    if len(cluster) == 0:
                        idx_empty_clusters.append(idx)

                if len(idx_empty_clusters) != 0:
                    freq_empty_clusters = [self.freq_cluster[x] for x in idx_empty_clusters]
                    rnum = random.uniform(0, sum(freq_empty_clusters))
                    for idx, freq in enumerate(freq_empty_clusters):
                        if rnum < freq:
                            extraction_list = [[x[0],x[1]] for x in self.original_clusters[idx_empty_clusters[idx]][1] if x not in self.banned_couples]


                            if len(extraction_list) == 0:
                                del self.couples_in_clusters[idx_empty_clusters[idx]]
                                del self.freq_cluster[idx_empty_clusters[idx]]
                                del self.original_clusters[idx_empty_clusters[idx]]
                                break
                            flag_cluster_found = True
                            self.trial_cluster = idx_empty_clusters[idx]
                            break
                        else:
                            rnum -= freq
                else:
                    rnum = random.uniform(0, sum(self.freq_cluster))
                    for idx, freq in enumerate(self.freq_cluster):
                        if rnum < freq:
                            extraction_list = [x for x in self.original_clusters[idx][1] if x not in self.banned_couples]

                            if len(extraction_list) == 0:
                                del self.original_clusters[idx]
                                del self.freq_cluster[idx]
                                break
                            flag_cluster_found = True  
                            self.trial_cluster = self.original_clusters[idx][0]                           
                            break
                        else:
                            rnum -= freq
                        
                if not flag_cluster_found:
                    continue
            
            else:
                extraction_list = [x for x in self.gen_couples if x not in self.banned_couples]
            
            causal_el = random.sample(extraction_list,1)[0]
            
            A=causal_el[0]
            B=causal_el[1]
            
            with open('log.txt','a') as fstdout:
                    fstdout.write(f'Trying generating new template with: {A+B} (Try #{count_flag})\n')
            copy_files(A, B, self.dir_all_poscars, self.dir_all_Individuals, self.comp)

            df_individuals = read_individuals(self.dir_all_Individuals+f'{A+B}_Individuals')
            P, SG = best_structures(df_individuals, 0.1, self.dir_all_poscars+f'{A+B}_gatheredPOSCARS')
                
            if len(SG) > 0:
                with open('log.txt','a') as fstdout:
                        fstdout.write(f'There are {len(SG)} possible templates: {SG}\n')
                self.trial_couple = causal_el
                StudiedSGs = []
                for k in range(len(SG)):
                    StudiedSGs.append(SG[k])
                    idx_couples = [i for i,x in enumerate(self.couples) if x[1] == SG[k]]
                    if self.is_not_in_couple(SG[k]):
                        self.trial_poscar = P[k]
                        self.trial_SG = int(SG[k])
                        break
                
                if self.trial_SG is None:
                    with open('log.txt','a') as fstdout:
                            fstdout.write(f'All possible templates simmetries are already chosen: {SG}\n')
                    StudiedSGs = []
                    for k in range(len(SG)):
                        StudiedSGs.append(SG[k])
                        
                        for l in self.couples:
                            template_already_chosen = False
                            if l[1] == SG[k] and l[0] == self.trial_couple:
                                with open('log.txt','a') as fstdout:
                                        fstdout.write(f'The couple has been already chosen with {l}\n')
                                template_already_chosen = True
                                break
                        
                        if not template_already_chosen:
                            self.trial_poscar = P[k]
                            self.trial_SG = int(SG[k])
                            break

                    if template_already_chosen:
                        self.banned_couples.append(self.trial_couple)
                        with open('log.txt','a') as fstdout:
                                fstdout.write(f'All the structure near ground state have been already chosen: the couple {self.trial_couple[0]+self.trial_couple[1]} won\'t be sorted again\n')
                        continue
                break   
            else:
                with open('log.txt','a') as fstdout:
                        fstdout.write(f'No structures with high simmetry near ground state: choosign new couple\n')
                self.banned_couples.append(causal_el)
                continue
        
        
        #c2q.poscar_to_input(str(self.trial_poscar), self.dir_all_cif + f'{self.num_template}.cif', self.dir_all_qeinput + f'{self.num_template}.in')
        with open('log.txt','a') as fstdout:
                fstdout.write(f'The couple {A+B} tries to make a structure with spacegroup {self.trial_SG} as the {self.num_template+1}-th template\n')
        
        return len(SG)
    
    def make_ranking_vec(self):
        self.flag_conv = True
        new_ranking = np.zeros((2, self.num_template))  
        A = self.trial_couple[0]
        B = self.trial_couple[1]

        for i in range(self.num_template):
        
            #with open('log.txt','a') as fstdout:
            #   fstdout.write(f'Running {i}-th relaxation for ranking vector for couple {A+B}\n')
            
            #overwrite_A, overwrite_B = mqe.find_element_type(self.dir_all_qeinput+f'{i}.in')
            #mqe.change_celldm1(self.dir_all_qeinput+f'{i}.in', overwrite_A, overwrite_B, A , B, self.comp )
        
            #mqe.setup_QEinput(self.dir_all_qeinput+f'{i}.in',self.dir_all_qeinput+f'{i}.in', str(Element(A)), str(Element(B)), float(Element(A).atomic_mass), float(Element(B).atomic_mass))
            #run_pw(self.dir_all_qeinput+f'{i}.in', self.dir_all_qeoutput+f'{i}.out', 4) 
        
            self.flag_conv =  mqe.check_convergence(self.dir_all_qeoutput + f'{i}.out')
            if not self.flag_conv:
                #with open('log.txt','a') as fstdout:
                #   fstdout.write(f'WARNING: {i}-th relaxation did not converge, skipped the couple {A+B}\n')
                self.n_fail+=1
                if self.trial_couple not in self.banned_couples:
                    self.banned_couples.append(self.trial_couple)
                new_ranking[:]=-1
                break

            if self.flag_conv:
                new_ranking[0,i]= mqe.find_enthalpy_relaxed(self.df,f'{self.couples[i][0][0]+self.couples[i][0][1]}', f'{self.trial_couple[0]+self.trial_couple[1]}', self.couples[i][1]) #find_enthalpy(self.dir_all_qeoutput+f'{i}.out')/find_natm(self.dir_all_qeoutput+f'{i}.out')                                           
                new_ranking[1,i] = int(i)    
                new_ranking = new_ranking[:, new_ranking[0].argsort()]   
        
        return new_ranking
    
    def own_relax(self, ranking_vec = np.array([[],[]]) ):
        A = self.trial_couple[0]
        B = self.trial_couple[1]

        #with open('log.txt','a') as fstdout:
        #   fstdout.write('Running relaxation on own template...\n')
        
        #mqe.setup_QEinput(self.dir_all_qeinput+f'{self.num_template}.in',self.dir_all_qeinput+f'{self.num_template}.in', str(Element(A)), str(Element(B)), float(Element(A).atomic_mass), float(Element(B).atomic_mass))
        #run_pw(self.dir_all_qeinput+f'{self.num_template}.in', self.dir_all_qeoutput+f'{self.num_template}.out', 4)
        new_fitness= mqe.find_enthalpy_relaxed(self.df,f'{self.trial_couple[0]+self.trial_couple[1]}', f'{self.trial_couple[0]+self.trial_couple[1]}', self.trial_SG) #find_enthalpy(self.dir_all_qeoutput+f'{self.num_template}.out')/find_natm(self.dir_all_qeoutput+f'{self.num_template}.out')
        new_ranking = np.append(ranking_vec, [[new_fitness], [self.num_template]], axis=1)

        if self.num_template == 0:
            self.data = np.zeros((2,1,1))
            self.data[0] = new_ranking[0]
            self.data[1] = new_ranking[1]

        return new_ranking

    def relax_on_new_template(self):
        self.flag_conv = True
        couples = self.couples
        num_template = self.num_template

        new_column = np.zeros(num_template)
        for i, couple in enumerate(couples):
            #with open('log.txt','a') as fstdout:
            #   fstdout.write(f'Running relaxation on new template for the {i}-th couple...\n')
            
            #overwrite_A, overwrite_B = find_element_type(self.dir_all_qeinput+f'{num_template}.in')
            #change_celldm1(self.dir_all_qeinput+f'{num_template}.in', overwrite_A, overwrite_B, str(Element(couple[0][0])) , str(Element(couple[0][1])), self.comp)
            #setup_QEinput(self.dir_all_qeinput+f'{num_template}.in',self.dir_all_qeinput+f'{num_template}.in', str(Element(couple[0][0])), str(Element(couple[0][1])), float(Element(couple[0][0]).atomic_mass), float(Element(couple[0][1]).atomic_mass))
            #run_pw(self.dir_all_qeinput+f'{num_template}.in', self.dir_all_qeoutput+f'{num_template}.out', 4)
            self.flag_conv =  mqe.check_convergence(self.dir_all_qeoutput + f'{i}.out')

            if not self.flag_conv:
                #with open('log.txt','a') as fstdout:
                #   fstdout.write(f'WARNING: {i}-th relaxation did not converge, skipped the couple {self.trial_couple[0]+self.trial_couple[1]}\n')
                new_column[:] = -1
                self.n_fails+=1
                if self.trial_couple not in self.banned_couples:
                    self.banned_couples.append(self.trial_couple)
                break
            if self.flag_conv:
                new_column[i] = mqe.find_enthalpy_relaxed(self.df,f'{self.trial_couple[0]+self.trial_couple[1]}', f'{couple[0][0]+couple[0][1]}', self.trial_SG) #find_enthalpy(self.dir_all_qeoutput+f'{num_template}.out')/find_natm(self.dir_all_qeoutput+f'{num_template}.out')
        if self.flag_conv:
            self.add_column(new_column)
        return new_column

    def add_column(self, column_values):
        # Add new column to the end of each matrix
        idx_column = np.full((self.data.shape[1]),self.num_template)
        self.data = np.pad(self.data, ((0, 0), (0, 0), (0, 1)), constant_values=0)
        self.data[0,:,-1] = column_values
        self.data[1,:,-1] = idx_column
        return

    def add_row(self, row_values, idx_row):
        # Add new row to the end of each matrix
        self.data = np.pad(self.data, ((0, 0), (0, 1), (0, 0)), constant_values=0)
        self.data[0,-1,:] = row_values
        self.data[1,-1,:] = idx_row
        return
    
    def order(self):
        # Order the matrix by rows of first matrix
        sorted_indices = np.argsort(self.data[0], axis=1)
        for i in range(self.data.shape[1]):
            self.data[0,i] = self.data[0,i, sorted_indices[i]]
            self.data[1,i] = self.data[1,i, sorted_indices[i]]
        return
    
    def distance(self, array):
        # Compute the Levenshtein distance between the rows of the first matrix and the array
        lev = np.zeros(self.data.shape[1])
        for i in range(self.data.shape[1]):
            lev[i] = levensthein_distance(self.data[1,i], array)
        return lev

    def update(self, n_possible_templates):
        new_tuple = (self.trial_couple, self.trial_SG)
        self.couples.append(new_tuple)
        self.couples_in_clusters[self.trial_cluster].append(new_tuple)
        self.poscars.append(self.trial_poscar)
        self.num_template += 1
        self.order()
        with open('log.txt','a') as fstdout:
                fstdout.write(f'The new template has been added to the set succesfully: {self.trial_couple[0]+self.trial_couple[1]} {self.trial_SG} \n')
        count = 0
        for i in self.couples:
            if i[0] == self.trial_couple:
                count+=1
        if count == n_possible_templates:
            with open('log.txt','a') as fstdout:
                    fstdout.write(f'All the structure near ground state have been already chosen: the couple {self.trial_couple[0]+self.trial_couple[1]} won\'t be sorted again\n')
            self.banned_couples.append(self.trial_couple)
        return 
    
    def recap_tempset(self, file = None):
        if file is not None:
            relax_file = file
        else:
            relax_file = f'TemplateSet_{self.comp}'
        with open(relax_file, 'w') as file:
            file.write(f'NUMBER OF TEMPLATES {self.num_template}\n')
            file.write(f'COMPOSITION A {self.comp} B\n')
            file.write('COUPLES , SPACEGROUPS \n')
            for i, couple in enumerate(self.couples):
                file.write(f'[{couple[0][0]} {couple[0][1]}] , {couple[1]} \n')    
            file.write('RANKING VECTORS ENTHALPIES\n')
            for i, vec in enumerate(self.data[0]):
                arr = str(vec).replace('\n','')
                file.write(f'{i}:{arr} \n')
            file.write('RANKING VECTORS IDX\n')
            for i, vec in enumerate(self.data[1]):
                arr = str(vec).replace('\n','')
                file.write(f'{i}:{arr} \n')  
            file.write('POSCARS\n')
            for i in self.poscars:
                file.write(str(i))
        return
class PairSet:
    def __init__(self, template_set, test_elements, relaxed_pairs = None, comp=1, mother_dir = './SETUP_FILES/', flag_temp_final = False  ) -> None:
        
        #DA METTERE INDIPENDENZA DA TEMPLATE SET
        self.test_elements=test_elements
        self.from_scratch = True   
        self.flag_temp_final = flag_temp_final
        self.num_pairs = 0
        self.n_fails = 0
        self.num_template = template_set.num_template
        self.comp = template_set.comp

        self.couples = []
        self.banned_couples = []
        self.poscars = template_set.poscars
        self.sg = []
        self.gen_couples = []
        for i in template_set.couples:
            self.sg.append(i[1])
            self.gen_couples.append(i[0])

        self.data = np.zeros((3, self.num_template, self.num_pairs))

        self.dir_all_cif = mother_dir + 'all_cif/'
        self.dir_all_qeoutput = mother_dir + 'all_qeoutput/'
        self.dir_all_qeinput = mother_dir + 'all_qeinput/'
        #create_directory(self.dir_all_cif)
        #create_directory(self.dir_all_qeoutput)
        #create_directory(self.dir_all_qeinput)

        self.df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'A{comp}B/relaxation/RELAX_DATA'), sep=",", index_col=0, na_filter = False)
        self.one_el = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'A{comp}B/relaxation/OneElementEnt.txt'), sep=',', header=None, na_filter=False)
        self.gs_df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'A{comp}B/relaxation/GroundStates.txt'), sep=",", na_filter = False)


        if relaxed_pairs is not None:
            self.from_scratch = False

            flag_couples=False
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
                    flag_couples = False                    
                if flag_couples:
                    self.couples.append(line[line.find('[')+1:line.find(']')].split())
                if line.startswith('RELAXED COUPLES'):
                    flag_couples = True      
        pass

    def add_pair(self):
        tries=0
        while True:
            if tries>=5:
                with open('log.txt','a') as fstdout:
                        fstdout.write('WARNING: too many tries, no new pair added\n')
                return
            tries+=1
            while True:
                causal_el = random.sample(self.test_elements, 2)
                if self.comp == 1:
                    causal_el.sort()
                if causal_el in self.banned_couples:
                    continue
                elif causal_el not in self.couples:
                    A=causal_el[0]
                    B=causal_el[1]
                    break
            with open('log.txt','a') as fstdout:
                    fstdout.write(f'Trying add to pair-set the couple {A+B} (Try#{tries})\n')

            new_ranking = np.zeros(self.num_template)
            for i in range(self.num_template):
                #with open('log.txt','a') as fstdout:
                #   fstdout.write(f'Running relaxation on {i}-th template with couple {A+B}\n')
                
                #overwrite_A, overwrite_B = mqe.find_element_type(self.dir_all_qeinput+f'{i}.in')
                #mqe.change_celldm1(self.dir_all_qeinput+f'{i}.in', overwrite_A, overwrite_B, A , B, self.comp)
                #mqe.setup_QEinput(self.dir_all_qeinput+f'{i}.in',self.dir_all_qeinput+f'{i}.in', str(Element(A)), str(Element(B)), float(Element(A).atomic_mass), float(Element(B).atomic_mass))
                #run_pw(self.dir_all_qeinput+f'{i}.in', self.dir_all_qeoutput+f'{i}.out', 4)
            
                convergence_flag =  mqe.check_convergence(self.dir_all_qeoutput + f'{i}.out')
                
                if not convergence_flag:
                    #with open('log.txt','a') as fstdout:
                    #   fstdout.write(f'WARNING: relaxation did not converge, skipped the couple {A+B}\n')
                    self.n_fails+=1
                    self.banned_couples.append([A,B])
                    break
                ent_form =  ( (self.one_el.loc[self.one_el[0] == f'{A}'].iloc[0,1]) * self.comp + (self.one_el.loc[self.one_el[0] == f'{B}'].iloc[0,1]) ) / (self.comp + 1 )
                new_ranking[i] = mqe.find_enthalpy_relaxed(self.df,f'{self.gen_couples[i][0]+self.gen_couples[i][1]}', f'{A+B}', self.sg[i]) - ent_form #new_ranking[i] = find_enthalpy(self.dir_all_qeoutput+f'{i}.out')/find_natm(self.dir_all_qeoutput+f'{i}.out') -ent_form
            if not convergence_flag:
                #with open('log.txt','a') as fstdout:
                #   fstdout.write('WARNING: Too many failed relaxations exiting...\n')
                return
            else:
                break

        self.data = np.pad(self.data, ((0, 0), (0, 0), (0, 1)), constant_values=0)
        self.data[0,:,-1] = new_ranking
        self.data[1,:,-1] = self.num_pairs
        self.data[2,:,-1] = np.arange(0, self.num_template, 1)
        self.couples.append([A,B])
        self.num_pairs += 1
        return
    
    def recap_relaxed(self, outfile = 'RelaxedPairs.txt'):
        with open(outfile, 'w') as file:
            file.write(f'NUMBER OF PAIRS {self.num_pairs}\n')
            file.write(f'COMPOSITION A {self.comp} B\n')
            file.write(f'NUMBER OF TEMPLATES {self.num_template}\n')
            file.write('RELAXED COUPLES\n')
            for i, couple in enumerate(self.couples):
                file.write(f'[{couple[0]} {couple[1]}] \n')  
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

    def order_wrt_pairs(self): #moves the pairs
        # Order the matrix by rows of first matrix
        matrix = self.data.copy()
        sorted_indices = np.argsort(matrix[0], axis=1)
        for i in range(matrix.shape[1]):
            matrix[0,i] = matrix[0,i, sorted_indices[i]]
            matrix[1,i] = matrix[1,i, sorted_indices[i]]
            matrix[2,i] = matrix[2,i, sorted_indices[i]]
        return matrix

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
        matrix = self.order_wrt_pairs()
        for i in range(self.num_template):
            for j in range(self.num_template):
                dist[i,j] = levensthein_distance(matrix[1,i], matrix[1,j])*2/(self.num_pairs+1)
        return dist
    
    def make_input(self):
        return
        # Make the input files for each template
        #for i in range(self.num_template):
            #poscar_to_input(str(self.poscars[i]), i, qe_input=qe_input)

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

    def reduced_set(self, hyperparameters):
        if not self.flag_temp_final:
            # Compute the reduced set of templates
            form_negative = self.formation_percentage()
            ist = self.template_gs()

            set_of_templates = [x for x in range(self.num_template)]
            set_of_removed_templates = []
            lev_matrix = self.dist_matrix()

            for i in range(self.num_template):
                if set_of_templates[i] in set_of_removed_templates:
                    continue
                for j in range(i+1,self.num_template):
                    if set_of_templates[j] in set_of_removed_templates:
                        continue
                    if lev_matrix[i,j] < hyperparameters['lev_red']:
                        a_j = form_negative[j] * hyperparameters['weight_formation_entalphy'] 
                        b_j = ist[j] * hyperparameters['weight_occurrence']/self.num_pairs
                        c_j = self.sg[j] * hyperparameters['weight_sg']

                        a_i = form_negative[i] * hyperparameters['weight_formation_entalphy']
                        b_i = ist[i] * hyperparameters['weight_occurrence']/self.num_pairs
                        c_i = self.sg[i] * hyperparameters['weight_sg']

                        score_j = a_j + b_j + c_j
                        score_i = a_i + b_i + c_i
                        
                        if score_j > score_i:
                            set_of_removed_templates.append(set_of_templates[i])
                            break
                        else:
                            set_of_removed_templates.append(set_of_templates[j])
            return [x for x in set_of_templates if x not in set_of_removed_templates]

        else:
            # Compute the reduced set of templates
            form_negative = self.formation_percentage()
            ist = self.template_gs()
            sg = self.sg.copy()

            set_of_templates = [x for x in range(self.num_template)]
            lev_matrix = self.dist_matrix()
            np.fill_diagonal(lev_matrix, 10)

            while len(set_of_templates) > hyperparameters['n_final_templates']:
                idx = np.unravel_index(np.argmin(lev_matrix), lev_matrix.shape)
                i = idx[0]
                j = idx[1]

                a_j = form_negative[j] * hyperparameters['weight_formation_entalphy'] 
                b_j = ist[j] * hyperparameters['weight_occurrence']/self.num_pairs
                c_j = sg[j] * hyperparameters['weight_sg']

                a_i = form_negative[i] * hyperparameters['weight_formation_entalphy']
                b_i = ist[i] * hyperparameters['weight_occurrence']/self.num_pairs
                c_i = sg[i] * hyperparameters['weight_sg']

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

    def error_single_composition(self, hyperparameters):
        # Compute the error of the single composition
        differences = []
        set_of_remaining_templates = self.reduced_set(hyperparameters)

        if self.comp == 1:
            for k in range(len(self.test_elements)):
                for l in range(k+1,len(self.test_elements)):
                
                    cp = [self.test_elements[k], self.test_elements[l]]
                    cp.sort()
                    try_couple = cp[0]+cp[1]

                    ent_gs = (self.gs_df.loc[self.gs_df['COUPLES'] == try_couple].iloc[0,1]) #already per atom  
                    ent_temp = np.zeros(len(set_of_remaining_templates))
                    for j, i in enumerate(set_of_remaining_templates):
                        ent_temp[j] = self.df.loc[try_couple, f'{self.gen_couples[i][0]}{self.gen_couples[i][1]}_{self.sg[i]}'] #already per atom
                    ent_temp.sort()
                    differences.append( -( ent_gs - ent_temp[0] )) 

        else:
            for k in range(len(self.test_elements)):
                for l in range(len(self.test_elements)):
                    if k == l:
                        continue
                    cp = [self.test_elements[k], self.test_elements[l]]
                    try_couple = cp[0]+cp[1]

                    ent_gs = (self.gs_df.loc[self.gs_df['COUPLES'] == try_couple].iloc[0,1]) #already per atom  
                    ent_temp = np.zeros(len(set_of_remaining_templates))
                    for j, i in enumerate(set_of_remaining_templates):
                        ent_temp[j] = self.df.loc[try_couple, f'{self.gen_couples[i][0]}{self.gen_couples[i][1]}_{self.sg[i]}'] #already per atom
                    ent_temp.sort()
                    differences.append( -( ent_gs - ent_temp[0] )) 
        return differences

    def total_error(self, hyperparameters):
        # Compute the total error of the reduced set
        differences = self.error_single_composition(hyperparameters)
        err = 0
        for i in differences:
            err += abs(i)/len(differences)
        return err
    

def generate_one_templateset(hyperparameters, test_elements, clusters):
    template = TemplateSet(test_elements=test_elements, comp = hyperparameters['comp'], clusters=clusters)

    tries = 0

    if template.from_scratch:
        # set the trial couple and return the number of possible templates remaining for that couple
        n_possible_temp = template.try_new_couple()
        template.own_relax()
        template.update(n_possible_temp)
        #with open('log.txt','a') as fstdout:
        #   fstdout.write(f'Class set inizialized: \n{template.data}\n')

    while template.num_template < hyperparameters['n_template']:
        if tries >= 10:
            with open('log.txt','a') as fstdout:
                fstdout.write(f'WARNING: too many tries, lowering lev thr from {hyperparameters["lev_gen"]} to {hyperparameters["lev_gen"]-0.1}\n')
            hyperparameters["lev_gen"] -= 0.1  
            tries = 0 
        tries += 1

        with open('log.txt','a') as fstdout:
            fstdout.write(f'Possible couples:{[x for x in template.gen_couples if x not in template.banned_couples]}\n')
            fstdout.write(f'Banned Couples:{template.banned_couples}\n')
            fstdout.write(f'Chosen Couples:{template.couples}\n')

        # Try a new couple
        n_possible_temp = template.try_new_couple()
        # Build the ranking vector for the new couple
        vec = template.make_ranking_vec()
        if not template.flag_conv:
            continue
        # Compute all the lev dist with other vectors
        lev_dist = template.distance(vec[1])*2/(template.num_template+1)
        with open('log.txt','a') as fstdout:
            fstdout.write(f'Try #{tries} \n Levenshtein distances: {lev_dist}\n')

        if np.any(lev_dist < hyperparameters["lev_gen"]) and template.num_template > 5:
            with open('log.txt','a') as fstdout:
                fstdout.write('Levensthein distance too low with some other template, trying new couple \n')
            continue

        vec = template.own_relax(vec)
        if not template.flag_conv:
            continue

        col = template.relax_on_new_template()
        if not template.flag_conv:
            continue

        template.add_row(*vec)
        template.update(n_possible_temp)
        tries = 0
    return template

def generate_one_pairset (template_prod, hyperparameters, test_elements):
    reduction_set = PairSet(template_prod, test_elements, comp=hyperparameters['comp'])
    reduction_set.make_input()

    while reduction_set.num_pairs < hyperparameters['n_pairs']:
        reduction_set.add_pair()

    return reduction_set


def graph_difference_std (dif_mean, dif_std, n_temp, c_value, dir_temp, hyperparameters, test_elements):
    fig, ax1 = plt.subplots(1,1, figsize=(25, 10))
    differences = dif_mean
    color_value = cm.viridis(c_value)

    ax1.bar(np.arange(0,len(differences),1), differences, yerr = dif_std , color=color_value, edgecolor='black', alpha=0.8)
    ax1.set_title(r'$\Delta H$ for each couple with '+f'{n_temp} templates')
    ax1.set_xticks(np.arange(0,len(differences),1))
    if hyperparameters['comp'] == 1:
        temp_ticks = []
        for i in range(len(test_elements)):
            for j in range(i+1,len(test_elements)):
                couple = [test_elements[i], test_elements[j]]
                couple.sort()
                temp_ticks.append(couple[0]+couple[1])

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


#Run pw.x in parallel using file input and output
def run_pw(fileinput, fileoutput, nproc):
    return
    commmand=f'mpirun -np {nproc} -x OMP_NUM_THREADS=1 pw.x < ' + fileinput + '>' + fileoutput
    mqe.set_elconvthr(fileinput, 1e-4)
    os.system(commmand)
    celldm_vec=mqe.find_celldm(fileoutput)
    mqe.set_celldm(fileinput, celldm_vec)
    mqe.set_elconvthr(fileinput, 1e-6)
    os.system(commmand)
    celldm_vec=mqe.find_celldm(fileoutput)
    mqe.set_elconvthr(fileinput, 1e-8)
    mqe.set_celldm(fileinput, celldm_vec)
    os.system(commmand)
    return
