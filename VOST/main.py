import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.feature_extraction import get_graph_node_names
from dataset import VODDataset
from torch.utils.data import Dataset, TensorDataset, DataLoader
from cnn_baseline import CNN_Network
from torchvision import transforms, models

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_features(data_loader, filename):
    all_features = []
    all_labels = []
    model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval()
    for i, l in data_loader:
        image = i.to(device)
        #print(image.shape) # is actually ([1, 3, 1080, 1920]
        #features = models.resnet50(image, weights=models.ResNet50_Weights.IMAGENET1K_V2) # batch, 2048
        with torch.no_grad():
            features = model(image)
            all_features.append(features.cpu())
            all_labels.append(l)
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    
    torch.save({
        "features": all_features,
        "labels": all_labels
    }, filename)
    

## TRAIN
def train(train_loader, model, optimiser, lossfn):
    model.train()
    for img, target in train_loader:
        images = img.to(device) #1,3,1080,1920
        optimiser.zero_grad()
        pred = model(images)
        #print("pred is", pred)
        #print("target is", target)
        loss = lossfn(pred, target)
        loss.backward()
        optimiser.step()

    print("final loss is ",  loss)

## TEST
def test(model, test_loader, lossfn):
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


if __name__ == '__main__':
    
    train_files =  [] # 572

    with open('ImageSets/train.txt', 'r') as fh:
        for line in fh:
            file = line.replace('\n', '')  
            train_files.append((file))
                
    # 70, 30 split
    ## can only do 30 videos at a time
    train_dataset = VODDataset(video_names=train_files[0:15])
    test_dataset = VODDataset(video_names=train_files[15:30])
    #train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False)
    train_feature_file_name = "resnet50_train_features.pt"
    test_feature_file_name =  "resnest50_test_features.pt"
    
    saving_time_start = time.time()
    if True:
        print("saving features") # when single vid 30 secs, now multiple 600 secs?!
        #save_features(train_loader, train_feature_file_name)
        save_features(test_loader, test_feature_file_name)
        print("saved")
    
    saving_time_end = time.time()
    print(f"Time taken to save train and test features is {saving_time_end-saving_time_start}")
        
    #After features are saved, load and train + test data
    data = torch.load(train_feature_file_name)
    features = data["features"]
    labels = data["labels"]
    
    #print(features.shape) # 5,2048
    #print(labels)#1,1,1,1
    
    new_train_dataset = TensorDataset(features, labels)
    new_train_loader = DataLoader(new_train_dataset, shuffle=True)
    
    data = torch.load(test_feature_file_name)
    features = data["features"]
    labels = data["labels"]
    
    new_test_dataset = TensorDataset(features, labels)
    new_test_loader = DataLoader(new_test_dataset, shuffle=True)
    
    #model = FasterRCNN(backbone=backbone, num_classes=1).to(device)
    model = CNN_Network(num_classes=30).to(device)
    lossfn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    training_start_time = time.time()
    train(new_train_loader, model, optimiser, lossfn)
    training_end_time = time.time()
    
    testing_start_time = time.time()
    test(model, new_test_loader, lossfn)
    testing_end_time = time.time()
    
    train_time = training_end_time - training_start_time
    test_time = testing_end_time - testing_start_time
    print(f"Train time is {train_time} and test time is {test_time}")