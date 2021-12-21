from molSimplify.Scripts.cellbuilder_tools import *
from molSimplify.Classes import mol3D
from molSimplify.Informatics.autocorrelation import*
from molSimplify.Informatics.misc_descriptors import*
from molSimplify.Informatics.graph_analyze import*
from molSimplify.Informatics.RACassemble import *
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import sparse
import itertools
from molSimplify.Informatics.MOF.PBC_functions import *
import networkx as nx

#### NOTE: In addition to molSimplify's dependencies, this portion requires 
#### pymatgen to be installed. The RACs are intended to be computed
#### on the primitive cell of the material. You can compute them
#### using the commented out snippet of code if necessary.

# Example usage is given at the bottom of the script.

'''<<<< CODE TO COMPUTE PRIMITIVE UNIT CELLS >>>>'''
#########################################################################################
# This MOF RAC generator assumes that pymatgen is installed.                            #
# Pymatgen is used to get the primitive cell.                                           #
#########################################################################################
from pymatgen.io.cif import CifParser
def get_primitive(datapath, writepath):
    s = CifParser(datapath, occupancy_tolerance=1).get_structures()[0]
    sprim = s.get_primitive_structure()
    sprim.to("cif",writepath)
'''<<<< END OF CODE TO COMPUTE PRIMITIVE UNIT CELLS >>>>'''

#########################################################################################
# The RAC functions here average over the different SBUs or linkers present. This is    #
# because one MOF could have multiple different linkers or multiple SBUs, and we need   #
# the vector to be of constant dimension so we can correlate the output property.       #
#########################################################################################

def make_MOF_SBU_RACs(SBUlist, SBU_subgraph, molcif, depth, name,cell,anchoring_atoms, sbupath=False, connections_list=False, connections_subgraphlist=False):
    n_sbu = len(SBUlist)
    print(SBUlist)
    G=nx.from_numpy_matrix(molcif.graph)
    cycles = nx.minimum_cycle_basis(G) # gets all closed rings in graph
    subcycle_list = []
    for cycle in cycles:
        skip_row = False
        for element in cycle:
            if molcif.getAtom(element).ismetal():
                skip_row = True
                break
        if not skip_row:
            subcycle_list.append(cycle)

    """""""""
    Loop over all SBUs as identified by subgraphs. Then create the mol3Ds for each SBU.
    """""""""
    for i, SBU in enumerate(SBUlist):
        atoms_in_sbu = []
        SBU_mol = mol3D()
        for val in SBU:
            atoms_in_sbu.append(val)
            SBU_mol.addAtom(molcif.getAtom(val))
        # SBU_mol.graph = SBU_subgraph[i].todense()
        
        """""""""
        For each linker connected to the SBU, find the lc atoms for the lc-RACs.
        """""""""
        for j, linker in enumerate(connections_list):
            if len(set(SBU).intersection(linker))>0:
                #### This means that the SBU and linker are connected.
                temp_mol = mol3D()
                link_list = []
                for jj, val2 in enumerate(linker):
                    if val2 in anchoring_atoms:
                        link_list.append(jj)
                    # This builds a mol object for the linker --> even though it is in the SBU section.
                    temp_mol.addAtom(molcif.getAtom(val2))

                temp_mol.graph = connections_subgraphlist[j].todense()
                """""""""
                If heteroatom functional groups exist (anything that is not C or H, so methyl is missed, also excludes anything lc, so carboxylic metal-coordinating oxygens skipped), 
                compile the list of atoms
                """""""""
                functional_atoms = []
                for jj in range(len(temp_mol.graph)):
                    if not jj in link_list:
                        if not set({temp_mol.atoms[jj].sym}) & set({"C","H"}):
                            functional_atoms.append(jj)
        # At this point, we look at the cycles for the graph, then add atoms if they are part of a cycle
        for cycle in subcycle_list:
            if (len(set(SBU).intersection(cycle))>0) and (len(set(SBU_mol.findMetal()).intersection(cycle))==0):
                for atom in cycle:
                    atoms_in_sbu.append(atom)
                    SBU_mol.addAtom(molcif.getAtom(atom))
                    for ringatom_connected_atoms in molcif.getBondedAtoms(atom):
                        if molcif.getAtom(int(ringatom_connected_atoms)).symbol()=='H':
                            atoms_in_sbu.append(ringatom_connected_atoms)
                            SBU_mol.addAtom(molcif.getAtom(ringatom_connected_atoms))

        # This part gets the subgraph and reassigns it, because we added atoms to the SBU
        tempgraph= molcif.graph[np.ix_(atoms_in_sbu,atoms_in_sbu)]
        SBU_mol.graph = tempgraph
        SBU_mol_cart_coords=np.array([atom.coords() for atom in  SBU_mol.atoms])
        SBU_mol_atom_labels=[atom.sym for atom in  SBU_mol.atoms]
        SBU_mol_adj_mat = np.array(SBU_mol.graph)
        ###### WRITE THE SBU MOL TO THE PLACE
        if sbupath and not os.path.exists(sbupath+"/"+str(name)+str(i)+'.xyz'):
            xyzname = sbupath+"/"+str(name)+"_sbu_"+str(i)+".xyz"
            SBU_mol_fcoords_connected = XYZ_connected(cell , SBU_mol_cart_coords , SBU_mol_adj_mat )
            writeXYZandGraph(xyzname , SBU_mol_atom_labels , cell , SBU_mol_fcoords_connected,SBU_mol_adj_mat)
    return None, None, None, None

def make_MOF_linker_RACs(linkerlist, linker_subgraphlist, molcif, depth, name, cell, linkerpath=False):
    #### This function makes full scope linker RACs for MOFs ####
    nlink = len(linkerlist)
    for i, linker in enumerate(linkerlist):
        linker_mol = mol3D()
        for val in linker:
            linker_mol.addAtom(molcif.getAtom(val))
        linker_mol.graph = linker_subgraphlist[i].todense()
        linker_mol_cart_coords=np.array([atom.coords() for atom in  linker_mol.atoms])
        linker_mol_atom_labels=[atom.sym for atom in  linker_mol.atoms]
        linker_mol_adj_mat = np.array(linker_mol.graph)
        ###### WRITE THE LINKER MOL TO THE PLACE
        if linkerpath and not os.path.exists(linkerpath+"/"+str(name)+str(i)+".xyz"):
            xyzname = linkerpath+"/"+str(name)+"_linker_"+str(i)+".xyz"
            linker_mol_fcoords_connected = XYZ_connected(cell, linker_mol_cart_coords, linker_mol_adj_mat)
            writeXYZandGraph(xyzname, linker_mol_atom_labels, cell, linker_mol_fcoords_connected, linker_mol_adj_mat)
    return None, None


def get_MOF_descriptors(data, depth, path=False, xyzpath = False):
    if not path:
        print('Need a directory to place all of the linker, SBU, and ligand objects. Exiting now.')
        raise ValueError('Base path must be specified in order to write descriptors.')
    else:
        if path.endswith('/'):
            path = path[:-1]
        if not os.path.isdir(path+'/ligands'):
            os.mkdir(path+'/ligands')
        if not os.path.isdir(path+'/linkers'):
            os.mkdir(path+'/linkers')
        if not os.path.isdir(path+'/sbus'):
            os.mkdir(path+'/sbus')
        if not os.path.isdir(path+'/xyz'):
            os.mkdir(path+'/xyz')
        if not os.path.isdir(path+'/logs'):
            os.mkdir(path+'/logs')
    ligandpath = path+'/ligands'
    linkerpath = path+'/linkers'
    sbupath = path+'/sbus'
    logpath = path+"/logs"

    """""""""
    Input cif file and get the cell parameters and adjacency matrix. If overlap, do not featurize.
    Simultaneously prepare mol3D class for MOF for future RAC featurization (molcif)
    """""""""

    cpar, allatomtypes, fcoords = readcif(data)
    cell_v = mkcell(cpar)
    cart_coords = fractional2cart(fcoords,cell_v)
    name = os.path.basename(data).strip(".cif")
    if len(cart_coords) > 2000:
        print("Too large cif file, skipping it for now...")
        tmpstr = "Failed to featurize %s: large primitive cell\n"%(name)
        write2file(path,"/FailedStructures.log",tmpstr)
        return None, None
    distance_mat = compute_distance_matrix2(cell_v,cart_coords)
    try:
        adj_matrix=compute_adj_matrix(distance_mat,allatomtypes)
    except NotImplementedError:
        tmpstr = "Failed to featurize %s: atomic overlap\n"%(name)
        write2file(path,"/FailedStructures.log",tmpstr)
        return None, None

    writeXYZandGraph(xyzpath, allatomtypes, cell_v, fcoords, adj_matrix.todense())
    molcif,_,_,_,_ = import_from_cif(data, True)
    molcif.graph = adj_matrix.todense()
    
    """""""""
    check number of connected components.
    if more than 1: it checks if the structure is interpenetrated. Fails if no metal in one of the connected components (identified by the graph).
    This includes floating solvent molecules.
    """""""""

    n_components, labels_components = sparse.csgraph.connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    metal_list = set([at for at in molcif.findMetal(transition_metals_only=False)])
    # print('##### METAL LIST', metal_list, [molcif.getAtom(val).symbol() for val in list(metal_list)])
    # print('##### METAL LIST', metal_list, [val.symbol() for val in molcif.atoms])
    if not len(metal_list) > 0:
        tmpstr = "Failed to featurize %s: no metal found\n"%(name)
        write2file(path,"/FailedStructures.log",tmpstr)
        return None, None

    for comp in range(n_components):
        inds_in_comp = [i for i in range(len(labels_components)) if labels_components[i]==comp]
        if not set(inds_in_comp)&metal_list:
            tmpstr = "Failed to featurize %s: solvent molecules\n"%(name)
            write2file(path,"/FailedStructures.log",tmpstr)
            return None, None

    if n_components > 1 :
        print("structure is interpenetrated")
        tmpstr = "%s found to be an interpenetrated structure\n"%(name)
        write2file(logpath,"/%s.log"%name,tmpstr)

    """""""""
    step 1: metallic part
        removelist = metals (1) + atoms only connected to metals (2) + H connected to (1+2)
        SBUlist = removelist + 1st coordination shell of the metals
    removelist = set()
    Logs the atom types of the connecting atoms to the metal in logpath.
    """""""""
    SBUlist = set() 
    metal_list = set([at for at in molcif.findMetal(transition_metals_only=False)])
    # print('##### METAL LIST2', metal_list, [molcif.getAtom(val).symbol() for val in list(metal_list)])
    # print('##### all LIST2', metal_list, [val.symbol() for val in molcif.atoms])
    [SBUlist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] #Remove all metals as part of the SBU
    [SBUlist.update(set(molcif.getBondedAtomsSmart(metal))) for metal in molcif.findMetal(transition_metals_only=False)]
    removelist = set()
    [removelist.update(set([metal])) for metal in molcif.findMetal(transition_metals_only=False)] #Remove all metals as part of the SBU
    for metal in removelist:
        bonded_atoms = set(molcif.getBondedAtomsSmart(metal))
        bonded_atoms_types = set([str(allatomtypes[at]) for at in set(molcif.getBondedAtomsSmart(metal))])
        cn = len(bonded_atoms)
        cn_atom = ",".join([at for at in bonded_atoms_types])
        tmpstr = "atom %i with type of %s found to have %i coordinates with atom types of %s\n"%(metal,allatomtypes[metal],cn,cn_atom)
        write2file(logpath,"/%s.log"%name,tmpstr)
    [removelist.update(set([atom])) for atom in SBUlist if all((molcif.getAtom(val).ismetal() or molcif.getAtom(val).symbol().upper() == 'H') for val in molcif.getBondedAtomsSmart(atom))] 
    """""""""
    adding hydrogens connected to atoms which are only connected to metals. In particular interstitial OH, like in UiO SBU.
    """""""""
    for atom in SBUlist:
        for val in molcif.getBondedAtomsSmart(atom):
            if molcif.getAtom(val).symbol().upper() == 'H':
               removelist.update(set([val])) 

    """""""""
    At this point:
    The remove list only removes metals and things ONLY connected to metals or hydrogens. 
    Thus the coordinating atoms are double counted in the linker.                         
    
    step 2: organic part
        removelist = linkers are all atoms - the removelist (assuming no bond between 
        organiclinkers)
    """""""""
    allatoms = set(range(0, adj_matrix.shape[0])) 
    linkers = allatoms - removelist
    linker_list, linker_subgraphlist = get_closed_subgraph(linkers.copy(), removelist.copy(), adj_matrix)
    connections_list = copy.deepcopy(linker_list)
    connections_subgraphlist = copy.deepcopy(linker_subgraphlist)
    linker_length_list = [len(linker_val) for linker_val in linker_list]
    adjmat = adj_matrix.todense()
    """""""""
    find all anchoring atoms on linkers and ligands (lc identification)
    """""""""
    anc_atoms = set()
    for linker in linker_list:
        for atom_linker in linker:
            bonded2atom  = np.nonzero(adj_matrix[atom_linker,:])[1] 
            if set(bonded2atom) & metal_list:
                anc_atoms.add(atom_linker)
    """""""""
    step 3: linker or ligand ?
    checking to find the anchors and #SBUs that are connected to an organic part
    anchor <= 1 -> ligand
    anchor > 1 and #SBU > 1 -> linker
    else: walk over the linker graph and count #crossing PBC
        if #crossing is odd -> linker
        else -> ligand
    """""""""
    initial_SBU_list, initial_SBU_subgraphlist = get_closed_subgraph(removelist.copy(), linkers.copy(), adj_matrix)
    templist = linker_list[:]
    tempgraphlist = linker_subgraphlist[:]
    long_ligands = False
    max_min_linker_length , min_max_linker_length = (0,100)
    for ii, atoms_list in reversed(list(enumerate(linker_list))): #Loop over all linker subgraphs
        linkeranchors_list = set() 
        linkeranchors_atoms = set() 
        sbuanchors_list = set() 
        sbu_connect_list = set()
        """""""""
        Here, we are trying to identify what is actually a linker and what is a ligand. 
        To do this, we check if something is connected to more than one SBU. Set to     
        handle cases where primitive cell is small, ambiguous cases are recorded.       
        """""""""
        for iii,atoms in enumerate(atoms_list): #loop over all atoms in a linker
            connected_atoms = np.nonzero(adj_matrix[atoms,:])[1] 
            for kk, sbu_atoms_list in enumerate(initial_SBU_list): #loop over all SBU subgraphs
                for sbu_atoms in sbu_atoms_list: #Loop over SBU
                    if sbu_atoms in connected_atoms:
                        linkeranchors_list.add(iii)
                        linkeranchors_atoms.add(atoms)
                        sbuanchors_list.add(sbu_atoms)
                        sbu_connect_list.add(kk) #Add if unique SBUs
        min_length,max_length = linker_length(linker_subgraphlist[ii].todense(),linkeranchors_list) 

        if len(linkeranchors_list) >=2 : # linker, and in one ambigous case, could be a ligand.
            if len(sbu_connect_list) >= 2: #Something that connects two SBUs is certain to be a linker
                max_min_linker_length = max(min_length,max_min_linker_length)
                min_max_linker_length = min(max_length,min_max_linker_length)
                continue
            else: 
                # check number of times we cross PBC :
                # TODO: we still can fail in multidentate ligands!
                linker_cart_coords=np.array([at.coords() \
                        for at in [molcif.getAtom(val) for val in atoms_list]])
                linker_adjmat = np.array(linker_subgraphlist[ii].todense())
                pr_image_organic = ligand_detect(cell_v,linker_cart_coords,linker_adjmat,linkeranchors_list)
                sbu_temp = linkeranchors_atoms.copy()
                sbu_temp.update({val for val in initial_SBU_list[list(sbu_connect_list)[0]]})
                sbu_temp = list(sbu_temp)
                sbu_cart_coords=np.array([at.coords() \
                       for at in [molcif.getAtom(val) for val in sbu_temp]])
                sbu_adjmat = slice_mat(adj_matrix.todense(),sbu_temp) 
                pr_image_sbu = ligand_detect(cell_v,sbu_cart_coords,sbu_adjmat,set(range(len(linkeranchors_list))))
                if not (len(np.unique(pr_image_sbu, axis=0))==1 and len(np.unique(pr_image_organic, axis=0))==1): # linker 
                    max_min_linker_length = max(min_length,max_min_linker_length)
                    min_max_linker_length = min(max_length,min_max_linker_length)
                    tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                            +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be linker\n'
                    write2file(ligandpath,"/ambiguous.txt",tmpstr)
                    continue
                else: #  all anchoring atoms are in the same unitcell -> ligand 
                    removelist.update(set(templist[ii])) # we also want to remove these ligands
                    SBUlist.update(set(templist[ii])) # we also want to remove these ligands
                    linker_list.pop(ii)
                    linker_subgraphlist.pop(ii)
                    tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
                            +','+' SBU connectlist: '+str(sbu_connect_list)+' set to be ligand\n'
                    write2file(ligandpath,"/ambiguous.txt",tmpstr)
                    tmpstr = str(name)+str(ii)+','+' Anchors list: '+ \
                            str(sbuanchors_list)+','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
                    write2file(ligandpath,"/ligand.txt",tmpstr)
        else: #definite ligand
            write2file(logpath,"/%s.log"%name,"found ligand\n")
            removelist.update(set(templist[ii])) # we also want to remove these ligands
            SBUlist.update(set(templist[ii])) # we also want to remove these ligands
            linker_list.pop(ii)
            linker_subgraphlist.pop(ii)
            tmpstr = str(name)+','+' Anchors list: '+str(sbuanchors_list) \
         +','+' SBU connectlist: '+str(sbu_connect_list)+'\n'
            write2file(ligandpath,"/ligand.txt",tmpstr)

    tmpstr = str(name) + ", (min_max_linker_length,max_min_linker_length): " + \
                str(min_max_linker_length) + " , " +str(max_min_linker_length) + "\n"
    write2file(logpath,"/%s.log"%name,tmpstr)
    if min_max_linker_length < 3: 
        write2file(linkerpath,"/short_ligands.txt",tmpstr)
    if min_max_linker_length > 2:
        # for N-C-C-N ligand ligand
        if max_min_linker_length == min_max_linker_length:
            long_ligands = True
        elif min_max_linker_length > 3:
            long_ligands = True
    
    """""""""
    In the case of long linkers, add second coordination shell without further checks. In the case of short linkers, start from metal
    and grow outwards using the include_extra_shells function
    """""""""
    linker_length_list = [len(linker_val) for linker_val in linker_list]
    if len(set(linker_length_list)) != 1:
        write2file(linkerpath,"/uneven.txt",str(name)+'\n')
    if not min_max_linker_length < 2: # treating the 2 atom ligands differently! Need caution
        if long_ligands:
            tmpstr = "\nStructure has LONG ligand\n\n"
            write2file(logpath,"/%s.log"%name,tmpstr)
            [[SBUlist.add(val) for val in  molcif.getBondedAtomsSmart(zero_first_shell)] for zero_first_shell in SBUlist.copy()] #First account for all of the carboxylic acid type linkers, add in the carbons.
        truncated_linkers = allatoms - SBUlist
        SBU_list, SBU_subgraphlist = get_closed_subgraph(SBUlist, truncated_linkers, adj_matrix)
        if not long_ligands:
            tmpstr = "\nStructure has SHORT ligand\n\n"
            write2file(logpath,"/%s.log"%name,tmpstr)
            SBU_list , SBU_subgraphlist = include_extra_shells(SBU_list,SBU_subgraphlist,molcif ,adj_matrix) 
    else:
        tmpstr = "Structure %s has extreamly short ligands, check the outputs\n"%name
        write2file(ligandpath,"/ambiguous.txt",tmpstr)
        tmpstr = "Structure has extreamly short ligands\n" 
        write2file(logpath,"/%s.log"%name,tmpstr)
        tmpstr = "Structure has extreamly short ligands\n" 
        write2file(logpath,"/%s.log"%name,tmpstr)
        truncated_linkers = allatoms - removelist
        SBU_list, SBU_subgraphlist = get_closed_subgraph(removelist, truncated_linkers, adj_matrix)
        SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list,SBU_subgraphlist,molcif ,adj_matrix)
        SBU_list, SBU_subgraphlist = include_extra_shells(SBU_list,SBU_subgraphlist,molcif ,adj_matrix)
    descriptor_names, descriptors, lc_descriptor_names, lc_descriptors = make_MOF_SBU_RACs(SBU_list, SBU_subgraphlist, molcif, depth, name , cell_v,anc_atoms, sbupath, connections_list, connections_subgraphlist)
    lig_descriptor_names, lig_descriptors = make_MOF_linker_RACs(linker_list, linker_subgraphlist, molcif, depth, name, cell_v, linkerpath)
    return None, None



