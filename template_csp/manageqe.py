import numpy as np
import re
import shutil
from pymatgen.core import Element

#Modify only the names of elements in atomic positions
def setup_QEinput(oldinput_path, newinput_path, A, B, mA, mB):
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
    return



#Check if the run is converged
def check_convergence(fileoutput_path):
    return True

    with open(fileoutput_path, 'r') as file:
        flag = False
        for line in file:
            if line.startswith('     Final scf calculation at the relaxed structure.'):
                flag = True
            if flag and line.startswith('     convergence has been achieved'):
                return True
            
        create_directory('./NotConverged')
        shutil.copyfile(fileoutput_path, f'./NotConverged/{fileoutput_path.split("/")[-1]}')

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

            if line.startswith('    celldm(4)'):
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
        for line in file:
            if line.startswith('     Final enthalpy'):
                enthalpy = re.search(r'-?\d+.\d+',line[line.find('=')+1:])

    return float(enthalpy.group())*13.6

#Find natm from output file of QE
def find_natm(fileoutput_path):

    with open(fileoutput_path, 'r') as file:
        for line in file:
            if line.startswith('     number of atoms/cell'):
                natm = re.search(r'\d+',line[line.find('='):])
                break
          
    return float(natm.group())



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