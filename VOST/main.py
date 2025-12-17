import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.feature_extraction import get_graph_node_names
from dataset import VODDataset
from torch.utils.data import Dataset, DataLoader
from cnn_network import CNN_Network

def masks_to_boxes(mask):
    boxes = []
    labels = []

    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0] ##
  
    for id in obj_ids:
        y, x = np.where(mask == id)
        if len(x) == 0:
            continue
        x1, y1, = x.min(), y.min()
        x2, y2 = x.max(), y.max()
        boxes.append([x1, y1, x2, y2])
        labels.append(int(id)) ## map to class label if needed
    return boxes, labels
    
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## NEXT: multiple videossss
train_dataset = VODDataset(video_name="0_squeeze_cloth", frames=(0,20))
test_dataset = VODDataset(video_name="0_squeeze_cloth", frames=(20,40))

train_loader = DataLoader(train_dataset, shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=False)
#model = FasterRCNN(backbone=backbone, num_classes=1).to(device)
model = CNN_Network(num_classes=3).to(device) # put .to(device)
lossfn = nn.CrossEntropyLoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

## TRAIN
model.train()
for img, target in train_loader:
    images = img.to(device)
    optimiser.zero_grad()
    pred = model(images)
    #print("pred is", pred)
    #print("target is", target)
    loss = lossfn(pred, target)
    loss.backward()
    optimiser.step()

print("final loss is ",  loss)

## TEST
model.eval()
size = len(test_loader.dataset)
num_batches = len(test_loader)
test_loss, correct = 0, 0

print("there are batches", num_batches)

with torch.no_grad():
    for img, target in test_loader:
        img = img.to(device)
        pred = model(img)
        #print("testing pred", pred)
        #print("testing target", target)
        test_loss += lossfn(pred, target).item()
        correct += (pred.argmax(1) == target).type(torch.float).sum().item()

test_loss /= num_batches
correct /= size
print(f"Accuracy is {(100*correct):>0.1f}%, Loss is {test_loss:>8f} \n")