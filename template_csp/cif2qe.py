import numpy as np
import re
from pymatgen.core import Element
from pymatgen.core import Composition
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter

AtomSymb = np.asarray([str(e) for e in Element])[2:]
AtomMass = np.asarray([str(float(Element(e).atomic_mass)) for e in AtomSymb]) 

bohr_to_angstrom = 0.52917720859 # Bohr radius
grk_pi = 3.14159265358979323846  # pi
rad_to_deg = 180 / grk_pi        # from radians to degrees

template_qe_input = {
"calculation_type" : "vc-relax", 
"pseudo_dir" : "./pseudo",
"pseudo_tail" : "_ONCV_PBE_sr.upf",
"ecutwfc" : 80,
"occupations_scf" : "smearing",
"smearing_scf" : "marzari-vanderbilt",
"degauss_scf" : 0.02,
"el_conv_thr" : 1.0e-7,
"mixing_beta" : 0.7,
"ion_dynamics" : "bfgs",
"cell_dofree" : "ibrav",
"cell_dynamics" : "bfgs",
"press" : 500.0,
"kspacing_scf" : 0.25,
"kspacing_nscf" : 0.10,
"forc_conv_thr" : 1e-3,
"etot_conv_thr" : 1e-5,
"cell_factor" : 4.0 
}

def make_kmesh(a, b, c, spacing = 0.30):
    KP_x = int(2.*grk_pi / (a * spacing) + 0.5)
    if KP_x < 1:
        KP_x = 1
    KP_y = int(2.*grk_pi / (b * spacing) + 0.5)
    if KP_y < 1:
        KP_y = 1
    KP_z = int(2.*grk_pi / (c * spacing) + 0.5)
    if KP_z < 1:
        KP_z = 1
    return [KP_x, KP_y, KP_z]  ## number of kpoints in each direction

# Given ibrav value and angle in rad returns the lattice vectors of primitive and conventional cell
def get_bravais_vectors(ibrav, alphar, betar, gammar):
    if ibrav == 1:
        ## 1: simple_cubic
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        return conventional, primitive
    elif ibrav == 2:
        ## 2: face_centered_cubic
        conventional =  np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[-1/2,0,1/2],
                                 [0,1/2,1/2],
                                 [-1/2,1/2,0]])
    elif ibrav == 3:
        ## 3: body_centered_cubic
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[1/2,1/2,1/2],
                                 [-1/2,1/2,1/2],
                                 [-1/2,-1/2,1/2]])
    elif ibrav == 4:
        ## 4: simple_hexagonal
        conventional = np.asarray([[1,0,0],
                                 [-1/2,np.sqrt(3)/2,0],
                                 [0,0,1]])
        primitive = np.asarray([[1,0,0],
                                 [-1/2,np.sqrt(3)/2,0],
                                 [0,0,1]])
    elif ibrav == 5:
        raise Warning("Trigonal axes not yet implemented.")
    elif ibrav == -5:
        raise Warning("Trigonal axes not yet implemented.")
    
    elif ibrav == 6:
    ## 6: simple_tetragonal
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
    elif ibrav == 7:
        ## 7: body_centered_tetragonal
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive =  np.asarray([[1/2,-1/2,1/2],
                                 [1/2,1/2,1/2],
                                 [-1/2,-1/2,1/2]])
    elif ibrav == 8:
        ## 8: orthorhombic_simple
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
    elif ibrav == 9:
        ## 9: orthorhombic_base_centered
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[1/2,1/2,0],
                                 [-1/2,1/2,0],
                                 [0,0,1]])
    elif ibrav == -9:
        ## -9: orthorhombic_base_centered
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[1/2,-1/2,0],
                                 [1/2,1/2,0],
                                 [0,0,1]])
    elif ibrav == 91:
     ## 91: orthorhombic_one_base_centered_a_type
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[1,0,0],
                                 [0,1/2,-1/2],
                                 [0,1/2,1/2]])
    elif ibrav == 10:
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[1/2,0,1/2],
                                 [1/2,1/2,0],
                                 [0,1/2,1/2]])
    elif ibrav == 11:
        ## 11: body_centered_orthorhombic
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [0,0,1]])
        primitive = np.asarray([[1/2,1/2,1/2],
                                 [-1/2,1/2,1/2],
                                 [-1/2,-1/2,1/2]])
    elif ibrav == 12:
        ## 12: monoclinic, unique axis c
        conventional = np.asarray([[1,0,0],
                                 [np.cos(gammar),0,np.sin(gammar)],
                                 [0, 0, 1]])
        primitive = np.asarray([[1,0,0],
                                 [np.cos(gammar),0,np.sin(gammar)],
                                 [0, 0, 1]])
    elif ibrav == -12:
        ## -12: monoclinic, unique axis b
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [np.cos(betar),0,np.sin(betar)]])
        primitive = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [np.cos(betar),0,np.sin(betar)]])
    elif ibrav == 13:
        ## 13: monoclinic, unique axis c
        conventional = np.asarray([[1,0,0],
                                 [np.cos(gammar),0,np.sin(gammar)],
                                 [0, 0, 1]])
        primitive = np.asarray([[1/2,0,-1/2],
                                 [np.cos(gammar),np.sin(gammar),0],
                                 [1/2, 0, 1/2]])
    elif ibrav == -13:
        ## -13: monoclinic base-centered, unique axis b
        conventional = np.asarray([[1,0,0],
                                 [0,1,0],
                                 [np.cos(betar),0,np.sin(betar)]])
        primitive = np.asarray([[1/2,1/2,0],
                                 [-1/2,1/2,0],
                                 [np.cos(betar),0,np.sin(betar)]])
    elif ibrav == 14:
        ## 14: triclinic, I hope this is never ever called
        conventional = np.asarray([[1,0,0],
                                 [np.cos(gammar), np.sin(gammar),0],
                                 [np.cos(betar),
                                  (np.cos(alphar)-np.cos(betar)*np.cos(gammar))/np.sin(gammar),
                                  np.sqrt(1+2*np.cos(alphar)*np.cos(betar)*np.cos(gammar)-np.cos(alphar)**2-np.cos(betar)**2-np.cos(gammar)**2)/np.sin(gammar)]])
        primitive = np.asarray([[1,0,0],
                                 [np.cos(gammar), np.sin(gammar),0],
                                 [np.cos(betar),
                                  (np.cos(alphar)-np.cos(betar)*np.cos(gammar))/np.sin(gammar),
                                  np.sqrt(1+2*np.cos(alphar)*np.cos(betar)*np.cos(gammar)-np.cos(alphar)**2-np.cos(betar)**2-np.cos(gammar)**2)/np.sin(gammar)]])
    else:
        raise Warning("No valid ibrav found, which should never happen, ibrav is: " + str(ibrav))
    return conventional, primitive

def check_and_add_position_primitive(new_coord, new_atom, coord_list, type_list, direct_lattice, threshold=0.001):
    list_operations = []
    for i in range(-3, 3, 1): ## vector a1 up to 4 cells away in all directions
        for j in range(-3, 3, 1): ## vector a2 up to 4 cells away in all directions
            for k in range(-3, 3, 1): ## vector a3 up to 4 cells away in all directions
                list_operations.append(direct_lattice[:,0]*i + direct_lattice[:,1]*j + direct_lattice[:,2]*k)

    for operation in list_operations:
        for old_coord in coord_list:
                ## compare element by element whether old_coord and new_coord + operation are all the same or not
                ## if they are the same, do not add this atom
            if all(abs(a - b) <= threshold for a, b in zip(np.asarray(old_coord), new_coord+np.asarray(operation))):
                return coord_list, type_list
    ## if the code arrives here it means that this atomic position is not equivalent to the previous ones
    coord_list.append(new_coord)
    type_list.append(new_atom)
    return coord_list, type_list

## write necessary celldm parameters for each ibrav
def write_celldm(file, qe_input, qe_parameters):
    if int(qe_parameters["ibrav"]) != 0:
        file.write("   ibrav = " + str(int(qe_parameters["ibrav"])) + " \n")
        file.write("   nat = " + str(int(qe_parameters["nat"])) + " \n")
        file.write("   ntyp = " + str(int(qe_parameters["ntyp"])) + " \n")
        file.write("   celldm(1) = " + str(qe_parameters["a"]/bohr_to_angstrom) + " \n")
        if qe_parameters["ibrav"] in [4, 6, 7]:
            file.write("   celldm(3) = " + str(qe_parameters["c"]/qe_parameters["a"]) + " \n")
        elif qe_parameters["ibrav"] in [5, -5]:
            file.write("   celldm(4) = " + str(np.cos(qe_parameters["alphar"])) + " \n")
        elif qe_parameters["ibrav"] == 14:
            file.write("   celldm(2) = " + str(qe_parameters["b"]/qe_parameters["a"]) + " \n")
            file.write("   celldm(3) = " + str(qe_parameters["c"]/qe_parameters["a"]) + " \n")
            file.write("   celldm(4) = " + str(np.cos(qe_parameters["alphar"])) + " \n")
            file.write("   celldm(5) = " + str(np.cos(qe_parameters["betar"])) + " \n")
            file.write("   celldm(6) = " + str(np.cos(qe_parameters["gammar"])) + " \n")
        elif qe_parameters["ibrav"] == -12:
            file.write("   celldm(2) = " + str(qe_parameters["b"]/qe_parameters["a"]) + " \n")
            file.write("   celldm(3) = " + str(qe_parameters["c"]/qe_parameters["a"]) + " \n")
            file.write("   celldm(5) = " + str(np.cos(qe_parameters["betar"])) + " \n")
        elif qe_parameters["ibrav"] == 13:
            file.write("   celldm(2) = " + str(qe_parameters["b"]/qe_parameters["a"]) + " \n")
            file.write("   celldm(3) = " + str(qe_parameters["c"]/qe_parameters["a"]) + " \n")
            file.write("   celldm(4) = " + str(np.cos(qe_parameters["alphar"])) + " \n")
        elif qe_parameters["ibrav"] in [8, 9, 10, 11]:
            file.write("   celldm(2) = " + str(qe_parameters["b"]/qe_parameters["a"]) + " \n")
            file.write("   celldm(3) = " + str(qe_parameters["c"]/qe_parameters["a"]) + " \n")
        elif qe_parameters["ibrav"] == 12:
            file.write("   celldm(2) = " + str(qe_parameters["b"]/qe_parameters["a"]) + " \n")
            file.write("   celldm(3) = " + str(qe_parameters["c"]/qe_parameters["a"]) + " \n")
            file.write("   celldm(4) = " + str(np.cos(qe_parameters["alphar"])) + " \n")
        elif qe_parameters["ibrav"] in [1, 2, 3]:
            pass

#Write base layer of the input file
def write_qe_input_lowlevel(file, qe_input, qe_parameters):
    ## Control   
    calculation_type = qe_input["calculation_type"]

    file.write("&CONTROL \n")
    file.write("   calculation = '" + calculation_type + "'\n")
    file.write("   restart_mode = 'from_scratch' \n")
    prefix = qe_parameters["chemical_formula"]
    file.write("   prefix = '" + prefix + "' \n")
    file.write("   tstress = .true. \n")
    file.write("   tprnfor = .true. \n")
    file.write("   pseudo_dir = '" + qe_input["pseudo_dir"] + "' \n")
    file.write("   outdir = './output' \n")
    file.write("   forc_conv_thr = " + str(qe_input["forc_conv_thr"]) + "\n")
    file.write("   etot_conv_thr = " + str(qe_input["etot_conv_thr"]) + "\n")
    file.write("&end \n")
    ## System
    file.write("&SYSTEM \n")
    write_celldm(file, qe_input, qe_parameters)
    file.write("   ecutwfc = " + str(qe_input["ecutwfc"]) + " \n")
    file.write("   occupations = '" + qe_input["occupations_scf"] + "' \n")
    file.write("   smearing = '" + qe_input["smearing_scf"] + "' \n")
    file.write("   degauss = " + str(qe_input["degauss_scf"]) + " \n")
    file.write("&end \n")
    ## Electrons
    file.write("&ELECTRONS \n")
    file.write("   conv_thr =  " + str(float(qe_input["el_conv_thr"])) + " \n")
    file.write("   mixing_beta = " + str(qe_input["mixing_beta"]) + " \n")
    file.write("&end \n")
    ## Ions
    file.write("&IONS \n")
    file.write("   ion_dynamics = '" + qe_input["ion_dynamics"] + "' \n")
    file.write("&end \n")
    ## Cell
    file.write("&CELL \n")
    file.write("   cell_dynamics = '" + qe_input["cell_dynamics"] + "' \n")
    file.write("   cell_dofree = '" + qe_input["cell_dofree"] + "' \n")
    #file.write("   cell_factor = " + str(float(qe_input["cell_factor"])) + " \n")
    file.write("   press = " + str(float(qe_input["press"])) + " \n")
    file.write("&end \n")
    file.write("\n")
    ## Atomic species
    file.write("ATOMIC_SPECIES \n")
    for i, specie in enumerate(qe_parameters["atomic_species"]):
        file.write("   " + specie + "   " + str(qe_parameters["atomic_masses"][i]) + "    " + specie + qe_input["pseudo_tail"] + " \n")
    file.write("\n")
    ## Atomic positions
    file.write("ATOMIC_POSITIONS crystal \n")
    for i, atom in enumerate(qe_parameters["atomic_coordinates"]):
        file.write("   " + qe_parameters["atomic_coordinates_types"][i] + "    " + 
                   str(qe_parameters["atomic_coordinates"][i][0]) + "    " + 
                   str(qe_parameters["atomic_coordinates"][i][1]) + "    " + 
                   str(qe_parameters["atomic_coordinates"][i][2]) + "    \n")
    ## K_POINTS
    file.write("K_POINTS automatic \n")
    if "kspacing_scf" in list(qe_input.keys()):
        kpoints = make_kmesh(qe_parameters["a"], qe_parameters["b"], qe_parameters["c"], spacing=qe_input["kspacing_scf"])
    else: 
        kpoints = make_kmesh(qe_parameters["a"], qe_parameters["b"], qe_parameters["c"], spacing=0.30)
    file.write(str(kpoints[0]) + "  " + str(kpoints[1]) + "  " + str(kpoints[2]) + "    0  0  0 " + " \n")

## Apply symmetry operation to a vector position to get a new vector position
def apply_symm_op(vec_pos, symm_op):
    x, y, z = vec_pos[0], vec_pos[1], vec_pos[2]
    new_vec_pos= [None, None, None]
    for i, coord in enumerate(symm_op):
        ## Replace symbolic coordinates with values
        op = symm_op[i].replace("x", "+"+str(x)).replace("y", "+"+str(y)).replace("z", "+"+str(z))
        ## Algebraic sign rules
        op = op.replace("--", "+").replace("-+", "-").replace("+-", "-").replace("++", "+")
        new_vec_pos[i] = float(eval(op))
        while new_vec_pos[i] < 0 :
            new_vec_pos[i] += 1.
        while new_vec_pos[i] >= 1. :
            new_vec_pos[i] -= 1.
    return new_vec_pos
    
## Function to identify Bravais lattice from lattice parameters
def find_lattice(a, b, c, alpha, beta, gamma):
    thr = 1.e-4
    bravais = ""
    if abs(alpha - 90.0) < thr and abs(gamma - 90.0) < thr:
        if abs(beta - 90.0) < thr:
            if abs(a - b) < thr and abs(a - c) < thr:
                bravais = "cubic"
            elif abs(a - b) < thr:
                bravais = "tetragonal"
            else:
                bravais = "orthorhombic"
        else:
            bravais = "monoclinic"
    elif abs(alpha - 90.0) < thr and abs(beta - 90.0) < thr and abs(gamma - 120.0) < thr:
        bravais = "hexagonal"
    elif abs(alpha - beta) < thr and abs(alpha - gamma) < thr and abs(a - b) < thr and abs(a - c) < thr:
        bravais = "rhombohedral"
    else:
        bravais = "triclinic"
    return bravais

## Function to find ibrav
def find_ibrav(spacegroup, bravais):
    ibrav = 0
    primitive = re.search("P", spacegroup) is not None
    bodycentered = re.search("I", spacegroup) is not None
    facecentered = re.search("F", spacegroup) is not None
    basecentered = re.search("C", spacegroup) is not None
    onefacebasecentered = re.search("A", spacegroup) is not None

    if bravais == "cubic":
        if primitive:
            ibrav = 1
        if facecentered:
            ibrav = 2
        if bodycentered:
            ibrav = 3
    elif bravais == "tetragonal":
        if primitive:
            ibrav = 6
        if bodycentered:
            ibrav = 7
    elif bravais == "orthorhombic":
        if primitive:
            ibrav = 8
        if basecentered:
            ibrav = 9
        if onefacebasecentered:
            ibrav = 91
        if facecentered:
            ibrav = 10
        if bodycentered:
            ibrav = 11
    elif bravais == "monoclinic":
        if primitive:
            ibrav = -12
        if basecentered:
            ibrav = 13
    elif bravais == "triclinic":
        ibrav = 14
    elif bravais == "hexagonal":
        ibrav = 4
    elif bravais == "rhombohedral":
        if primitive:
            ibrav = 4
        else:
            ibrav = 5
    else:
        ibrav = 0
    return ibrav

class PWSCf_input:
    def __init__(self, qe_input, cifname=None):
        self.name = None   ## name/identifier for structure
        self.cifname = cifname
        self.qe_input = qe_input
        ## Tolerance for recognizing identical atoms generated by symmetry
        self.threshold = 0.01
        self.totatom = 0
        self.num_symm_op = 0
        ## Store all symmetry operations
        self.symm_op_list = []
        ## Store atomic coordinates (pre sym-op)
        self.atomic_type_list = []
        self.atomic_coord_list = []
        self.qe_parameters = {}
        ## Store atomic coordinates and atom type (post sym-op)
        self.atomic_coord_list_extended = []
        self.atomic_type_list_extended = []
        ## Store list of atomic masses
        self.atomic_masses = []
        ## Internal consistency checks
        self.relax_read = False
        ## Relaxation checks
        self.off_diagonal_ok = None
        self.restart_diff_ok = None
        self.restart_forces_ok = None

    ## Reading keywords in cif file and add them to qe_parameters dictionary
    def read_cif(self):
        ## Token to count lines with symmetry operations
        count_symm_finished = False
        start_match = False
        ## Token to read lines with atomic positions
        start_readatoms = False
        with open(self.cifname) as cif_file:
            for line in cif_file.readlines():
                if "_symmetry_space_group_name_H-M" in line:
                    tmpspacegroup = line.split()[1]
                    self.qe_parameters["HMsg"] = tmpspacegroup
                elif "_cell_length_a" in line:
                    a = float(line.split()[1])
                    self.qe_parameters["a"] = a
                elif "_cell_length_b" in line:
                    b = float(line.split()[1])
                    self.qe_parameters["b"] = b
                elif "_cell_length_c" in line:
                    c = float(line.split()[1])
                    self.qe_parameters["c"] = c
                elif "_cell_angle_alpha" in line:
                    alpha = float(line.split()[1])
                    self.qe_parameters["alpha"] = alpha
                    self.qe_parameters["alphar"] = alpha / 180.0 * grk_pi
                    self.qe_parameters["cosab"] = np.cos(alpha / 180.0 * grk_pi)
                elif "_cell_angle_beta" in line:
                    beta = float(line.split()[1])
                    self.qe_parameters["beta"] = beta
                    self.qe_parameters["betar"] = beta / 180.0 * grk_pi
                    self.qe_parameters["cosbc"] = np.cos(beta / 180.0 * grk_pi)
                elif "_cell_angle_gamma" in line:
                    gamma = float(line.split()[1])
                    self.qe_parameters["gamma"] = gamma
                    self.qe_parameters["gammar"] = gamma / 180.0 * grk_pi
                    self.qe_parameters["cosac"] = np.cos(gamma / 180.0 * grk_pi)
                elif "_symmetry_Int_Tables_number" in line:
                    tmptablenumber = int(line.split()[1])
                    self.qe_parameters["NUMsg"] = tmptablenumber
                elif "_chemical_formula_structural" in line:
                    chemical_formula = line.split()[1]
                    self.qe_parameters["chemical_formula"] = chemical_formula
                elif "_chemical_formula_sum" in line:
                    chemical_formula_sum = line.replace("'","").split()[1:]
                    self.qe_parameters["formula_sum"] = chemical_formula_sum
                ## Count symmetry operations
                elif " _symmetry_equiv_pos_as_xyz" in line:
                    start_match = True
                elif start_match and not count_symm_finished:
                    self.num_symm_op += 1
                if ("loop_" in line) and start_match:
                    count_symm_finished = True
                    start_match = False
                    self.num_symm_op -= 1
                if start_match and self.num_symm_op > 0 :
                    self.symm_op_list.append(line.split("'")[1].replace("'","").replace(",","").split())
                ## Count symmetry operation ends
                ## Read atom type and coordinate begins
                if start_readatoms and line.strip():
                    self.atomic_coord_list.append(line.split()[3:6])
                    self.atomic_type_list.append(line.split()[0])
                if "_atom_site_occupancy" in line:
                    start_readatoms = True
                ## Read atom type and coordinate ends
        self.data_read = True
        self.generate_cell_data()
        self.generate_coords_sym()
        self.get_unique_atoms_list()
        self.name = self.qe_parameters["chemical_formula"] + "_" + str(self.qe_parameters["NUMsg"])

    ## Write the input file for Quantum Espresso
    def write_scf_input(self, qe_input_name="pw.scf.in"):
        if self.data_read:
            with open(qe_input_name, "w") as qe_input_file:
                write_qe_input_lowlevel(qe_input_file, self.qe_input, self.qe_parameters)
        else:
            with open('log.txt','a') as fstdout:
                    fstdout.write("Data was not read!")

    ## Get the name of atoms without repetition
    def get_unique_atoms_list(self):
        _, idx = np.unique(np.asarray(self.atomic_type_list_extended), return_index=True)
        self.qe_parameters["atomic_species"] = np.asarray(self.atomic_type_list_extended)[np.sort(idx)].tolist()
        for element in self.qe_parameters["atomic_species"]:
            self.atomic_masses.append(float(AtomMass[np.isin(AtomSymb, element)][0]))
        self.qe_parameters["atomic_masses"] = self.atomic_masses  

    def generate_coords_sym(self):
        if self.data_read:
            for a_i, atomic_coord in enumerate(self.atomic_coord_list):
                for symm_op in self.symm_op_list:
                    ## generate a new equivalent position using the allowed symmetry operations
                    eq_coord = apply_symm_op(atomic_coord, symm_op)
                    ## apply conversion from conventional to primitive basis
                    ## convert the coordinates from conventional to primitive using the transform matrices
                    ## defined from the qe convention. Note that the inverse of the transform matrix is used
                    ## because coordinates transform inversely w.r.t. lattice vectors
                    conventional, primitive = get_bravais_vectors(self.qe_parameters["ibrav"], self.qe_parameters["alphar"],
                                self.qe_parameters["betar"],self.qe_parameters["gammar"])
                    transform_matrix_prim_to_conv = conventional @ np.linalg.inv(primitive)
                    eq_coord_prim = (transform_matrix_prim_to_conv.T @ np.asarray(eq_coord).T).T
                    ## here the check for repetition is done already in the primitive cell
                    ## if the position was not present, it is added to self.atomic_coord_list_extended
                    check_and_add_position_primitive(eq_coord_prim, self.atomic_type_list[a_i], 
                    self.atomic_coord_list_extended, self.atomic_type_list_extended, 
                    primitive, threshold=self.threshold)
            ##

            self.qe_parameters["atomic_coordinates"] = self.atomic_coord_list_extended
            self.qe_parameters["atomic_coordinates_types"] = self.atomic_type_list_extended
            ## Number of atoms from composition
            self.nat = len(self.atomic_coord_list_extended)
            self.qe_parameters["nat"] = int(self.nat)

    ## Angles in radiants
    def generate_cell_data(self):
        ## Finding Bravais lattice and corresponding ibrav
        lattice = find_lattice(self.qe_parameters["a"], self.qe_parameters["b"], self.qe_parameters["c"], 
            self.qe_parameters["alpha"], self.qe_parameters["beta"], self.qe_parameters["gamma"])
        self.ibrav = find_ibrav(self.qe_parameters["HMsg"], lattice)
        ## Composition shenanigans
        self.composition = Composition(" ".join(self.qe_parameters["formula_sum"]))
        ## Number of type of atoms from composition
        self.ntyp = len(self.composition.formula.split(" "))

        self.qe_parameters["ibrav"] = self.ibrav
        self.qe_parameters["ntyp"] = self.ntyp


#Needs in input the name of the poscar and the name of the cif in output     
def poscar_to_cif(poscar, cif, symprec, angprec):
    poscar = Poscar.from_str(poscar, read_velocities=True)
    try:
        cif_structure = CifWriter(poscar.structure, symprec=symprec, angle_tolerance=angprec)
    except:
        with open('Errors.txt','a') as fstdout:
                fstdout.write("Pymatgen returned an error. Your symprec might be too large")
    cif_structure.write_file(cif)
    return

#Take the poscar in str format and write the qe input
def poscar_to_input(poscar_str, cif_path, input_path, qe_input = template_qe_input, symprec=0.2, angprec=3):
    poscar_to_cif(poscar_str, cif_path, symprec, angprec)
    qe = PWSCf_input(qe_input, cifname=cif_path)
    qe.read_cif()
    qe.write_scf_input(qe_input_name=input_path)
    return