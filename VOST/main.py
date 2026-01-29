import os
import cv2
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.feature_extraction import get_graph_node_names
from dataset import VODDataset
from torch.utils.data import Dataset, TensorDataset, DataLoader
from cnn_baseline import CNN_Classifier
from torchvision import transforms, models

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_features(data_loader, filename):
    """ 
        Input: 
            data_loader : frames from videos
            filename : feature file name for saving
    
        Passes frames through resnet50. Then aggregates frames (mean) and stores the video features alongside the label ID
    """
    model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval()
    video_to_frame_features = {}
    for i, l in data_loader:
        if l.item() not in video_to_frame_features.keys():
            video_to_frame_features[l.item()] = []
        image = i.to(device)
        with torch.no_grad():
            features = model(image)
            video_to_frame_features[l.item()].append(features.cpu())

    aggregated_video_features = []
    aggregated_video_labels=[]
    
    for label, frames in video_to_frame_features.items():
        averaged_features = np.mean(frames, axis=0) # aggregated frame features
        aggregated_video_features.append(torch.from_numpy(averaged_features))
        aggregated_video_labels.append(torch.tensor([label]))
        

    torch.save({
        "features": torch.cat(aggregated_video_features), # number of vids x 2048
        "labels": torch.cat(aggregated_video_labels)
    }, filename)
    

## TRAIN
def train(train_loader, model, optimiser, lossfn):
    model.train()
    for img, target in train_loader:
        images = img.to(device) # Video feature input with dim 1x2028
        optimiser.zero_grad()
        pred = model(images)
        loss = lossfn(pred, target)
        loss.backward()
        optimiser.step()

    print("final loss is ",  loss)

## TEST
def test(model, test_loader, lossfn, train_loader):
    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    print(f"there are {num_batches} batches")

    with torch.no_grad():
        for img, target in test_loader:
            img = img.to(device)
            pred = model(img)
            test_loss += lossfn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Accuracy is {(100*correct):>0.1f}%, Loss is {test_loss:>8f} \n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--ExtractFeatures", type=str,
                        required=False)
    args = parser.parse_args()
    extract_features = bool(args.ExtractFeatures)
    
    train_files =  [] # 572

    # Loading video names from train.txt
    with open('ImageSets/train.txt', 'r') as fh:
        for line in fh:
            file = line.replace('\n', '')
            train_files.append((file))

    number_of_classes = 1        
    train_feature_file_name = f"{number_of_classes}_resnet50_train_features.pt" # 30_ for 30 videos
    test_feature_file_name =  f"{number_of_classes}_resnest50_test_features.pt"
    training_frame_range = (0,5)
    testing_frame_range = (5,15)

    if extract_features:
        saving_time_start = time.time()
        train_dataset = VODDataset(video_names=train_files[0:number_of_classes], frames=training_frame_range)
        test_dataset = VODDataset(video_names=train_files[0:number_of_classes], frames=testing_frame_range)
        train_loader = DataLoader(train_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=False)
        save_features(train_loader, train_feature_file_name)
        save_features(test_loader, test_feature_file_name)
        saving_time_end = time.time()
        print(f"Time taken to save train and test features is {saving_time_end-saving_time_start}")
        
    #After features are saved, load and train + test data
    train_video_features = torch.load(train_feature_file_name)
    features = train_video_features["features"]
    labels = train_video_features["labels"]
    new_train_dataset = TensorDataset(features, labels)
    new_train_loader = DataLoader(new_train_dataset, batch_size=16, shuffle=True)
    
    test_video_features = torch.load(test_feature_file_name)
    features = test_video_features["features"]
    labels = test_video_features["labels"]
    new_test_dataset = TensorDataset(features, labels)
    new_test_loader = DataLoader(new_test_dataset, batch_size=32, shuffle=False)
    
    model = CNN_Classifier(num_classes=number_of_classes).to(device)
    lossfn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    training_start_time = time.time()
    train(new_train_loader, model, optimiser, lossfn)
    training_end_time = time.time()
    
    testing_start_time = time.time()
    test(model, new_test_loader, lossfn, new_train_loader)
    testing_end_time = time.time()
    
    train_time = training_end_time - training_start_time
    test_time = testing_end_time - testing_start_time
    print(f"Train time is {train_time} and test time is {test_time}")