from torch_geometric.data import Dataset,DataLoader
import os.path as osp
import torch
from utils import unique_atom_list,data_frame_to_list
from rdkit.Chem import PandasTools

batch_size = 64


dir_train = r'C:\Users\zcemg08\PycharmProjects\graph_nn\data\amide_class_train_lbl.sdf'
dir_test  = r'C:\Users\zcemg08\PycharmProjects\graph_nn\data\amide_class_test_lbl.sdf'

amide_train     = PandasTools.LoadSDF(dir_train,smilesName='SMILES',molColName='Molecule',\
                                      includeFingerprints=False)
amide_test      = PandasTools.LoadSDF(dir_test, smilesName='SMILES',molColName='Molecule',\
                                      includeFingerprints=False)

'''If test atoms not in train --> problem '''

_, atom_dict = unique_atom_list(amide_train)

train   = DataLoader(data_frame_to_list(amide_train,atom_dict),batch_size=batch_size,shuffle=True)
test    = DataLoader(data_frame_to_list(amide_test,atom_dict),batch_size=batch_size,shuffle=True)


for batch in train:
    break

print(batch.y)

