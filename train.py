from torch_geometric.data import DataLoader
from rdkit.Chem import PandasTools
import os.path as osp
import torch

from utils import unique_atom_list,data_frame_to_list
from model import GCN



batch_size = 64


dir_train = r'C:\Users\zcemg08\PycharmProjects\graph_nn\data\amide_class_train_lbl.sdf'
dir_test  = r'C:\Users\zcemg08\PycharmProjects\graph_nn\data\amide_class_test_lbl.sdf'

amide_train     = PandasTools.LoadSDF(dir_train,smilesName='SMILES',molColName='Molecule',\
                                      includeFingerprints=False)
amide_test      = PandasTools.LoadSDF(dir_test, smilesName='SMILES',molColName='Molecule',\
                                      includeFingerprints=False)

'''If test atoms not in train --> problem '''

_, atom_dict = unique_atom_list(amide_train)

train_loader   = DataLoader(data_frame_to_list(amide_train,atom_dict),batch_size=batch_size,shuffle=True)
test_loader    = DataLoader(data_frame_to_list(amide_test,atom_dict),batch_size=batch_size,shuffle=True)


model = GCN(num_node_features=13,hidden_channels=64,num_classes=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train(loader):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.



def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 201):
    train(train_loader)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')



