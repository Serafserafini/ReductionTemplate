import numpy as np
import re
from pymatgen.core import Element
from pymatgen.io.vasp import Poscar
from pymatgen.core.structure import Structure

#Modify only the names of elements in atomic positions
def setup_QEinput(oldinput_path, newinput_path, A, B):
    mA, mB = float(Element(A).atomic_mass), float(Element(B).atomic_mass)
    old_el=[]
    new_el=[A,B]
    check = False
    with open(oldinput_path, 'r') as file:
        text_input=file.readlines()
    
    for i, line in enumerate(text_input):
        if line.startswith('   prefix'):
            text_input[i]=f'   prefix = "{A}{B}"\n'            
            
        if line.startswith('ATOMIC_SPECIES'):
            old_el.append(text_input[i+1].split()[0])
            old_el.append(text_input[i+2].split()[0])

            text_input[i+1]=f'{A} {mA} {A}_ONCV_PBE_sr.upf\n'
            text_input[i+2]=f'{B} {mB} {B}_ONCV_PBE_sr.upf\n'
        if line.startswith('K_POINTS'):
            check = False
        if check:
            atom=line.split()[0]
            text_input[i]=line.replace(atom,new_el[old_el.index(atom)])
        if line.startswith('ATOMIC_POSITIONS'):
            check = True
     
    with open(newinput_path, 'w') as file:
        file.writelines(text_input)
    return old_el[0], old_el[1]



#Check if the run is converged
def check_convergence(fileoutput_path):

    with open(fileoutput_path, 'r') as file:
        flag = False
        for line in file:
            if line.startswith('     Final scf calculation at the relaxed structure.'):
                flag = True
            if flag and line.startswith('     convergence has been achieved'):
                return True
            
        # create_directory('./NotConverged')
        # shutil.copyfile(fileoutput_path, f'./NotConverged/{fileoutput_path.split("/")[-1]}')

    return False

#Set the celldm in the input file
def set_celldm(fileinput_path, celldm_vec):

    with open(fileinput_path, 'r') as file:
        text_input=file.readlines()
    for i, line in enumerate(text_input):
        if line.startswith('   celldm(1)'):
            text_input[i]=f'   celldm(1) = {celldm_vec[0]}\n'
        if line.startswith('   celldm(2)'):
            text_input[i]=f'   celldm(2) = {celldm_vec[1]}\n'
        if line.startswith('   celldm(3)'):
            text_input[i]=f'   celldm(3) = {celldm_vec[2]}\n'
        if line.startswith('   celldm(4)'):
            text_input[i]=f'   celldm(4) = {celldm_vec[3]}\n'
        if line.startswith('   celldm(5)'):
            text_input[i]=f'   celldm(5) = {celldm_vec[4]}\n'
        if line.startswith('   celldm(6)'):
            text_input[i]=f'   celldm(6) = {celldm_vec[5]}\n'
    with open(fileinput_path, 'w') as file:
        file.writelines(text_input)
    return

#set the elconvthr in the input file
def set_elconvthr(fileinput_path, elconvthr):

    with open(fileinput_path, 'r') as file:
        text_input=file.readlines()
    for i, line in enumerate(text_input):
        if line.startswith('   conv_thr'):
            text_input[i]=f'   conv_thr =  {elconvthr}\n'
    with open(fileinput_path, 'w') as file:
        
        file.writelines(text_input)
    return

#Change the celldm1 in the input file in order to make more easy the vc relax
def change_celldm1(file_input, A_old, B_old, A_new, B_new, comp): 
    A_old_r= float(Element(A_old).atomic_radius)
    B_old_r= float(Element(B_old).atomic_radius)
    A_new_r= float(Element(A_new).atomic_radius)
    B_new_r= float(Element(B_new).atomic_radius)    

    with open(file_input, 'r') as file:
        text_input=file.readlines()
    for i, line in enumerate(text_input):
        if line.startswith('   celldm(1)'):
            celldm_old=float(re.search(r'-?\d+.\d+',line).group())
            celldm_new = celldm_old * (A_new_r * comp + B_new_r ) / ( A_old_r * comp +B_old_r )
            text_input[i]='   celldm(1) = ' + str(celldm_new) + ' \n'
            break
    with open(file_input, 'w') as file:
        file.writelines(text_input)
    return   


#Needs output file of QE and return array of celldm
def find_celldm(fileoutput_path):
    celldm=np.full(6,0.0)
    
    with open(fileoutput_path, 'r') as file:
        for line in file:                                               #find i valori di partenza dei celldm
            if line.startswith('     celldm(1)'):
                dm = (re.search(r'celldm\(1\)=\s*-?\d+.\d+',line)).group()
                celldm[0] = float(dm[dm.find('=') + 1 : ])

                dm = (re.search(r'celldm\(2\)=\s*-?\d+.\d+',line)).group()
                celldm[1] = float(dm[dm.find('=') + 1 : ])
                
                dm = (re.search(r'celldm\(3\)=\s*-?\d+.\d+',line)).group()
                celldm[2] = float(dm[dm.find('=') + 1 : ])

            if line.startswith('     celldm(4)'):
                dm = (re.search(r'celldm\(4\)=\s*-?\d+.\d+',line)).group()
                celldm[3] = float(dm[dm.find('=') + 1 : ])

                dm = (re.search(r'celldm\(5\)=\s*-?\d+.\d+',line)).group()
                celldm[4] = float(dm[dm.find('=') + 1 : ])

                dm = (re.search(r'celldm\(6\)=\s*-?\d+.\d+',line)).group()
                celldm[5] = float(dm[dm.find('=') + 1 : ])                    
                
                break
    with open(fileoutput_path, 'r') as file:
        for i in file:   
            if i.startswith(' celldm'):
                dm = (re.search(r'celldm\(\d\) =\s*-?\d+.\d+',i)).group()
                celldm[int(dm[7])-1] = float(dm[dm.find('=') + 1 : ])                
    return celldm



#Find enthalpy from output file of QE
def find_enthalpy(fileoutput_path):
    
    with open(fileoutput_path, 'r') as file:
        lines = file.readlines()
        for line in lines[::-1]:
            if line.startswith('     Final enthalpy'):
                enthalpy = re.search(r'-?\d+.\d+',line[line.find('=')+1:])
                break

    return float(enthalpy.group())*13.6

#Find natm from output file of QE
def find_natm(fileoutput_path):

    with open(fileoutput_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('     number of atoms/cell'):
                natm = re.search(r'\d+',line[line.find('='):])
                return float(natm.group())
    
    return None



#Find element type in the qe input file (for change in celldm1)
def find_element_type(file_input):
    with open(file_input) as file:
        text_input = file.readlines()

    for i, line in enumerate(text_input):
        if line.startswith('ATOMIC_SPECIES'):
            A = text_input[i+1].split()[0]
            B = text_input[i+2].split()[0]
            break
    return A, B

#Find Enthalpy from previous relaxed
def find_enthalpy_relaxed(df, gen_couple, rel_couple, sg):
    enthalpy = df.loc[rel_couple, f'{gen_couple}_{sg}']
    return enthalpy

# Find the final pressure from the output file
def find_final_P(file_out_path):
    with open(file_out_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines[::-1]:
        if line.startswith('          total   stress  (Ry/bohr**3)'):
            return float(line.split('=')[1])
    
    return None

# Find the final volume from the output file
def find_volume(file_out_path):
    with open(file_out_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines[::-1]:
        if line.startswith('     new unit-cell volume'):
            return float(re.search(r'(\d+\.\d*) Ang\^3', line).group(1))
    
    return None

def check_good_conv(file_out_path):
    
    vol = find_volume(file_out_path)
    P = abs(find_final_P(file_out_path)-500)
    nat = find_natm(file_out_path)

    if vol is None or P is None or nat is None:
        return False

    if vol*P*6.242e-4/nat < 0.1:
        return True
    else:
        return False
    
def find_inputcelldm(file_input_path):
    cell_dm = np.full(6,0.0)
    with open(file_input_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'celldm' in line:
            cell_dm[int(line.split('(')[1][0])-1] = float(line.split('=')[1])
    return cell_dm


    

### IN CASE OF IBRAV = 0

def find_CELLPARAMETERS(path_file_out):
    cell_parameters = []

    with open(path_file_out) as file_out:
        lines = file_out.readlines()
        for line_id, line in enumerate(lines[::-1]):
            if line.startswith("CELL_PARAMETERS (angstrom)"):
                for i in range(3):
                    cell_parameters.append(list(map(float,lines[-line_id + i ].split())))
                break

    return cell_parameters

def find_ATOMICPOS(path_file_out):
    atomic_pos = []

    with open(path_file_out) as file_out:
        lines = file_out.readlines()

    natm = find_natm(path_file_out)
    if natm == None:
        return atomic_pos

    for line_id, line in enumerate(lines[::-1]):
        if line.startswith("ATOMIC_POSITIONS (crystal)"):
            for i in range(natm):
                Atom = lines[-line_id + i].split()[0]
                atomic_pos.append((Atom , list(map(float,lines[-line_id + i].split()[1:]))))
            break

    return atomic_pos

def set_CELLPARAMETERS(path_file_in, cell_parameters):
    if not cell_parameters:
        return

    with open(path_file_in) as file_in:
        lines = file_in.readlines()
    for line_id, line in enumerate(lines):
        if line.startswith("CELL_PARAMETERS"):
            for i in range(3):
                lines[line_id + i + 1] = " ".join(map(str,cell_parameters[i])) + "\n"
            break

    with open(path_file_in, "w") as file_in:
        file_in.writelines(lines)
    return
        
def set_ATOMICPOS(path_file_in, atomic_pos):
    if not atomic_pos:
        return

    with open(path_file_in) as file_in:
        lines = file_in.readlines()
    for line_id, line in enumerate(lines):
        if 'nat' in line:
            lines[line_id] = '   nat = ' + str(len(atomic_pos)) + '\n'
        if line.startswith("ATOMIC_POSITIONS"):
            for i in range(len(atomic_pos)):
                lines[line_id + i + 1] = atomic_pos[i][0] + " " + " ".join(map(str,atomic_pos[i][1])) + "\n"
            break
    
    with open(path_file_in, "w") as file_in:
        file_in.writelines(lines)
    return

def find_nbfgs(path_file_out):
    with open(path_file_out) as f:
        lines = f.readlines()
    bfgs = -1
    for line in lines[::-1]:
        if line.startswith('     number of bfgs steps'):
            bfgs = int(line.split('=')[1])
            break
    return bfgs

    
def find_trustradius(file_out):
    with open(file_out, 'r') as f:
        lines = f.readlines()
    last_trust_radius = -1
    for line in lines[::-1]:
        if line.startswith('     new trust radius '):
            last_trust_radius = float(re.search(r'(\d+\.\d+)', line).group(1))
            break
    return last_trust_radius

def new_celldm_find(poscar_path, oldA, oldB, A, B):
    with open(poscar_path, 'r') as f:
        text = f.read()
        poscar = Poscar.from_str(text)
        struttura = poscar.structure

    matrice_distanze = struttura.distance_matrix
    matrice_distanze[np.eye(len(matrice_distanze), dtype=bool)] = np.inf
    flat_indices = np.argsort(matrice_distanze, axis=None)
    row_indices, col_indices = np.unravel_index(flat_indices, matrice_distanze.shape)
    
    distanze_minime = matrice_distanze[row_indices, col_indices]
    specie_minime = [
        (struttura[row_indices[i]].specie, struttura[col_indices[i]].specie)
        for i in range(matrice_distanze.size)
    ]
    
    proportions = []
    prop_min = []

    for i in range(matrice_distanze.size):
        distanza_minima = distanze_minime[i]
        minA = specie_minime[i][0]
        minB = specie_minime[i][1]
        newA = A if str(minA) == oldA else B
        newB = B if str(minB) == oldB else A
        
        r_atomA = float(Element(newA).atomic_radius) if not Element(newA).atomic_radius == None else 0.0
        r_newA = min( float(Element(newA).average_ionic_radius), r_atomA ) 
        if r_newA == 0:
            r_newA = max( float(Element(newA).average_ionic_radius), r_atomA)
        if r_newA == 0:
            r_newA = float(Element(newA).atomic_radius_calculated) if not Element(newA).atomic_radius_calculated == None else 2.0

        r_atomB = float(Element(newB).atomic_radius) if not Element(newB).atomic_radius == None else 0.0
        r_newB = min( float(Element(newB).average_ionic_radius), r_atomB )
        if r_newB == 0:
            r_newB = max( float(Element(newB).average_ionic_radius), r_atomB)
        if r_newB == 0:
            r_newB = float(Element(newB).atomic_radius_calculated) if not Element(newB).atomic_radius_calculated == None else 2.0
        
        r_atomminA = float(Element(minA).atomic_radius) if not Element(minA).atomic_radius == None else 0.0
        r_minA = min( float(Element(minA).average_ionic_radius), r_atomminA )
        if r_minA == 0:
            r_minA = max( float(Element(minA).average_ionic_radius), r_atomminA)
        if r_minA == 0:
            r_minA = float(Element(minA).atomic_radius_calculated) if not Element(minA).atomic_radius_calculated == None else 2.0
        
        r_atomminB = float(Element(minB).atomic_radius) if not Element(minB).atomic_radius == None else 0.0
        r_minB = min( float(Element(minB).average_ionic_radius), r_atomminB )
        if r_minB == 0:
            r_minB = max( float(Element(minB).average_ionic_radius), r_atomminB)
        if r_minB == 0:
            r_minB = float(Element(minB).atomic_radius_calculated) if not Element(minB).atomic_radius_calculated == None else 2.0
        
        new_mean_r = (r_newA + r_newB) / 2
        min_mean_r = (r_minA + r_minB) / 2
        legame_min = max(r_newA + r_newB, 1.1)

        legame_max = r_newA + r_newB + 0.3
        
        proportions.append(new_mean_r / min_mean_r)
        proportions.append(legame_max / distanza_minima)

        prop_min.append(legame_min / distanza_minima)
    
    prop_min_acceptable = max(prop_min)

    len_in = len(proportions)
    proportions = [prop for prop in proportions if prop >= prop_min_acceptable]
    len_fin = len(proportions)
    if len_fin < len_in:
        return prop_min_acceptable
    else:
        return min(proportions)
