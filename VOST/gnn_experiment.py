import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import torch_geometric
from torch import Tensor
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import save_region_features
from torchvision import transforms, models
from torchvision.models import resnet50
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from cnn_baseline import CNN_Classifier
from torch_geometric.loader import DataLoader
from skimage.measure import label, regionprops
from gnn_dataset import GNN_Dataset
from torchvision.models.detection.roi_heads import roi_align

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class St_GCN_Classifier(torch.nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        # model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # for name, layer in model.named_children():
        #     if name in ['conv1', 'bn1', 'layer1', 'layer2']:
        #         for param in layer.parameters():
        #             param.requires_grad = False
            

        self.hidden_channels = 64
        self.conv1 = GCNConv(2048,self.hidden_channels) # input feature per node = 2, output feature per node =4
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.classifier = nn.Linear(self.hidden_channels, num_classes)

    def forward(self,x, edge_index, batch):

        # Node level message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph level pooling
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


# video = "0_squeeze_cloth"
video = "1214_cut_mellon"
video_graph = GNN_Dataset(video_names=["1214_cut_mellon"])
new_test_loader = DataLoader(video_graph, batch_size=16, shuffle=False)
v = video_graph.__getitem__(0)[0]

dataset = [v] # graph of images not features
cnn_dataset = []
train_loader = DataLoader(dataset, batch_size=64, shuffle = True)

gnn_model = St_GCN_Classifier(num_classes=2).to(device)
cnn_model = CNN_Classifier(num_classes=1).to(device) # TODO: fix num_classes, get unique object/action labels based on videos
gnn_optimiser = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
cnn_optimiser = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

gnn_model.train()
cnn_model.train()
for epoch in range(50):
    total_loss = 0

    for batch in train_loader: # for frame, graph
        batch = batch.to(device)

        gnn_optimiser.zero_grad()
        #cnn_optimiser.zero_grad()

        #cnn_pred = cnn_model() # input the frame
        gnn_pred = gnn_model(batch.x, batch.edge_index, batch.batch)

        gnn_loss = criterion(gnn_pred, batch.y)
        #cnn_loss = criterion(cnn_pred, batch.y)
        
        #all_cnn_loss_per_videos += cnn_loss
        
        gnn_loss.backward()
        gnn_optimiser.step()

        total_gnn_loss += gnn_loss.item()

    print(f"Epoch {epoch:03d}, GNN Loss: {total_gnn_loss:.4f}")


# model = MyGCN()
# out = model(video_1_graph)
# print("Output ndoe features after GN layer is", out)