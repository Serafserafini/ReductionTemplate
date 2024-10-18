import pandas as pd
import shutil
import os
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher


#Needs in input the name of the individuals file and saves it in a pandas df
def read_individuals(individuals):
    column_names = {0 : "Generation", 1 : "ID", 2 : "GenMode", 
    4 : 'A', 5 : 'B', 7 : "enthalpy", 10 : "fitness", 15 : "spacegroup"}

    individuals = pd.read_csv(individuals, sep="\s+", header=None, skiprows=2,usecols=column_names)
    individuals.rename(columns=column_names, inplace=True)
    individuals.sort_values("fitness", inplace=True)
    return individuals

#Needs in input the name of the file with all the poscars, the id of the structure to be extracted and the name of the output poscar
def find_poscar(gatheredPOSCARS_path, id):
    end=-1
    with open(gatheredPOSCARS_path,'r') as file:
        testo_input = file.readlines()              
    for i, line in enumerate(testo_input):
        if line.startswith('EA'+str(id)):   
            init = i
            simm=int(line[line.find(':')+1:])
        if line.startswith('EA'+str(id+1)):
            end = i-1
            break
    if end == -1:                            
        end = len(testo_input)-1
        
    poscar_str=''
    for i in range(end-init+1):
        poscar_str+=testo_input[init+i]
    return poscar_str

#Needs df of Individuals, fitness treshold, file gatheredPoscars and return best non duplicated structures (with symm>75)
def best_structures(individuals_df, fitness_upto, gatheredPOSCARS_path):
    fitness_gs = individuals_df['fitness'].iloc[0]
    uniques = []
    SGs=[]
    structure_gs = Poscar.from_str(find_poscar(gatheredPOSCARS_path, individuals_df['ID'].iloc[0]))
    
    if individuals_df['spacegroup'].iloc[0] > 75:
        uniques.append(structure_gs)
        SGs.append(individuals_df['spacegroup'].iloc[0])

    for i, line_individuals_df in individuals_df.iterrows():
        if line_individuals_df['fitness'] - fitness_gs >= fitness_upto:
            break
        if line_individuals_df['spacegroup'] < 75:
            continue
        new_structure = Poscar.from_str(find_poscar(gatheredPOSCARS_path, line_individuals_df['ID']))

        check_dupilcate = False
        
        if StructureMatcher(ltol = 1.0, stol = 1.0, angle_tol = 10, scale=True).fit(structure_gs.structure, new_structure.structure):
            check_dupilcate = True 
        
        for structure in uniques:
            if StructureMatcher(ltol = 1.0, stol = 1.0, angle_tol = 10, scale=True).fit(structure.structure, new_structure.structure):
                check_dupilcate = True
        if not check_dupilcate:
            uniques.append(new_structure)
            SGs.append(line_individuals_df['spacegroup'])
    return uniques, SGs


#Copy Individuals and gatheredPoscars files 
def copy_files(A, B, dir_all_poscars, dir_all_Individuals, comp):
    try:
        shutil.copyfile(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'A{comp}B/OUTPUTFILES/{A+B}/{A+B}_gatheredPOSCARS'), dir_all_poscars+f'{A+B}_gatheredPOSCARS')
        shutil.copyfile(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'A{comp}B/OUTPUTFILES/{A+B}/{A+B}_Individuals'), dir_all_Individuals+f'{A+B}_Individuals')
    except:
        with open('Errors.txt', 'a') as file:
            file.write(f'Error in copying {A+B} files: remember to put the files in the A{comp}B/OUTPUTFILES/{A+B}/ directory\n')
        raise
    return

