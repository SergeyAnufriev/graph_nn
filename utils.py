from rdkit import Chem
import torch
from torch_geometric.data import Data


def unique_atom_list(df):

    '''Find unique atom types in dataset and
    return dict for atom type'''

    atom_list = []

    for i in range(len(df)):
        for atom in df.iloc[i]['Molecule'].GetAtoms():
            n = atom.GetAtomicNum()
            if n not in atom_list:
                atom_list.append(n)

    atom_list = sorted(atom_list)

    return atom_list, {x:i for i,x in enumerate(atom_list)}


def get_node_type(mol,encode_dict):

    '''Mol --> one hot atom type matrix'''
    nodes = []

    for atom in mol.GetAtoms():
        x = torch.zeros(len(encode_dict))
        x[encode_dict[atom.GetAtomicNum()]] = 1
        nodes.append(x)

    return torch.vstack(nodes).type(torch.LongTensor)


def get_edge(mol):

    '''Mol ---> adjecency and bond feature matrix'''

    edge_dict = {Chem.rdchem.BondType.SINGLE:0,\
             Chem.rdchem.BondType.DOUBLE:1,\
             Chem.rdchem.BondType.TRIPLE:2,\
             Chem.rdchem.BondType.AROMATIC:3}

    row, col = [], []
    bonds    = []

    for bond in mol.GetBonds():

        '''Compute adjecency matrix in COO format'''
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]

        '''Find bond type'''
        x = torch.zeros(4)
        x[edge_dict[bond.GetBondType()]] = 1
        bonds.append(x)

    return torch.tensor([row, col], dtype=torch.long), torch.vstack(bonds).type(torch.LongTensor)



def mol_to_pytorch(mol,y,atom_dict):

    '''mol and traget y to pytorch geometric object'''

    node_features             = get_node_type(mol,atom_dict)

    edge_index, bond_features = get_edge(mol)

    data = Data(x=node_features, edge_index=edge_index, \
                edge_attr=bond_features, y=torch.LongTensor([y]))

    return data


def data_frame_to_list(df,atom_dict):

    '''dataframe to list of DATA pytorch geometric objects'''
    data_list = []
    for _,row in df.iterrows():

        data_list.append(mol_to_pytorch(row['Molecule'],int(row['activity']),atom_dict))

    return data_list

