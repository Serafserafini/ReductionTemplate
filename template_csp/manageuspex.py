import pandas as pd
import shutil
import os
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher


#Needs in input the name of the individuals file and saves it in a pandas df
def read_individuals(individuals_path):
    """
    Reads the Individuals file from USPEX and returns a DataFrame with relevant columns:
    Generation, ID, GenMode, A, B, enthalpy, fitness, spacegroup.
    The lines are sorted by fitness in ascending order.

    Args:
        individuals_path (str): Path to the Individuals file.
    Returns:
        individuals_df (pd.DataFrame): DataFrame containing the relevant columns from the Individuals file.
    """

    column_names = {0 : "Generation", 1 : "ID", 2 : "GenMode", 
    4 : 'A', 5 : 'B', 7 : "enthalpy", 10 : "fitness", 15 : "spacegroup"}

    individuals_df = pd.read_csv(individuals_path, sep=r"\s+", header=None, skiprows=2,usecols=column_names)
    individuals_df.rename(columns=column_names, inplace=True)
    individuals_df.sort_values("fitness", inplace=True)
    return individuals_df

#Needs in input the name of the file with all the poscars, the id of the structure to be extracted and the name of the output poscar
def find_poscar(gatheredPOSCARS_path, id):

    """
    Finds the POSCAR string for a given structure ID from the gathered POSCARS file of USPEX.
    Args:
        gatheredPOSCARS_path (str): Path to the gathered POSCARs file.
        id (int): ID of the structure to find.
    Returns:
        poscar_str (str): The POSCAR string corresponding to the given ID.
    """

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
def best_structures(individuals_df, fitness_upto, gatheredPOSCARS_path, sg_min_acceptable=75):

    """
    Takes a DataFrame of Individuals file from USPEX, a fitness threshold, the path to gathered POSCAR files and a minimum acceptable space group.
    It returns a list of unique structures and their corresponding space groups, ordered by enthalpy.
    The function first checks the fitness of the first individual and then iterates through the DataFrame to find unique structures that meet the fitness criteria.
    It uses the StructureMatcher from pymatgen to check for duplicates based on the structure's symmetry and scale.
    The function also filters out structures with a space group below the specified minimum acceptable value.

    Args:
        individuals_df (pd.DataFrame): DataFrame containing Individuals file from USPEX.
        fitness_upto (float): Fitness threshold to filter individuals.
        gatheredPOSCARS_path (str): Path to the gathered POSCAR files.
        sg_min_acceptable (int): Minimum acceptable space group for structures.
    
    Returns:
        uniques (list): List of unique structures that meet the fitness criteria.
        SGs (list): List of space groups corresponding to the unique structures.
    """


    fitness_gs = individuals_df['fitness'].iloc[0]
    uniques = []
    SGs=[]
    structure_gs = Poscar.from_str(find_poscar(gatheredPOSCARS_path, individuals_df['ID'].iloc[0]))
    
    if individuals_df['spacegroup'].iloc[0] > sg_min_acceptable:
        uniques.append(structure_gs)
        SGs.append(individuals_df['spacegroup'].iloc[0])

    for i, line_individuals_df in individuals_df.iterrows():
        if line_individuals_df['fitness'] - fitness_gs >= fitness_upto:
            break
        if line_individuals_df['spacegroup'] < sg_min_acceptable:
            continue
        new_structure = Poscar.from_str(find_poscar(gatheredPOSCARS_path, line_individuals_df['ID']))

        check_duplicate = False
        for idx, structure in enumerate(uniques):
            if StructureMatcher(ltol = 1.0, stol = 1.0, angle_tol = 10, scale=True).fit(structure.structure, new_structure.structure):
                check_duplicate = True
                break

        if not check_duplicate:
            uniques.append(new_structure)
            SGs.append(line_individuals_df['spacegroup'])

    return uniques, SGs


def groundstate_structure(individuals_df, gatheredPOSCARS_path):
    """
    Takes a DataFrame of Individuals file from USPEX and the path to gathered POSCAR files.
    Return the poscar of the ground state structure.

    Args:
        individuals_df (pd.DataFrame): DataFrame containing Individuals file from USPEX.
        gatheredPOSCARS_path (str): Path to the gathered POSCAR files.
    Returns:
        structure_gs (Poscar): The POSCAR object of the ground state structure.
    """
    structure_gs = Poscar.from_str(find_poscar(gatheredPOSCARS_path, individuals_df['ID'].iloc[0]))
    return structure_gs


#Copy Individuals and gatheredPoscars files 
def copy_files(A, B, dir_all_poscars, dir_all_Individuals, comp):
    """
    Copies the gathered POSCARS and Individuals files from the USPEX output directory to the specified directories.
    Args:
        A (str): The first element of the system.
        B (str): The second element of the system.
        dir_all_poscars (str): Directory to copy the gathered POSCARS files to.
        dir_all_Individuals (str): Directory to copy the Individuals files to.
        comp (str): The composition of the system.
    """
    
    try:
        shutil.copyfile(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'A{comp}B/{A+B}/{A+B}_gatheredPOSCARS'), dir_all_poscars+f'{A+B}_gatheredPOSCARS')
        shutil.copyfile(os.path.join(os.path.dirname(os.path.dirname(__file__)),f'A{comp}B/{A+B}/{A+B}_Individuals'), dir_all_Individuals+f'{A+B}_Individuals')

    except Exception as e:
        with open('Errors.txt', 'a') as file:
            file.write(f'Error in copying {A+B} files: {str(e)} - remember to put the files in the A{comp}B/OUTPUTFILES/{A+B}/ directory\n')
        raise
    return

