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
from utils import get_file_names, get_unique_labels
from torchvision import transforms, models

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_features(data_loader, filename, aggregate=False):
    """ 
        Input: 
            data_loader : frames from videos
            filename : feature file name for saving
    
        Passes frames through resnet50. Then aggregates frames (mean) and stores the video features alongside the label ID
    """
    model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval()
    final_labels = []
    final_features = []
    video_to_frame_features = {}
    for i, l in data_loader:
        if l.item() not in video_to_frame_features.keys():
            video_to_frame_features[l.item()] = []
        image = i.to(device)
        with torch.no_grad():
            extracted_features = model(image)
            video_to_frame_features[l.item()].append(extracted_features.cpu())
            final_labels.append(l)
            final_features.append(extracted_features.cpu())

    if aggregate:
        aggregated_video_features = []
        aggregated_video_labels=[]
        
        for label, frames in video_to_frame_features.items():
            averaged_features = np.mean(frames, axis=0) # aggregated frame features
            aggregated_video_features.append(torch.from_numpy(averaged_features))
            aggregated_video_labels.append(torch.tensor([label]))
        
        final_labels = aggregated_video_labels
        final_features = aggregated_video_features
        
    torch.save({
        "features": torch.cat(final_features), # number of vids x 2048
        "labels": torch.cat(final_labels)
    }, filename)
    

## TRAIN
def train(train_loader, model, optimiser, lossfn):
    model.train()
    for img, target in train_loader:
        images = img.to(device) # Video feature input with dim 1x2028
        optimiser.zero_grad()
        pred = model(images)
        # predicted_class = torch.argmax(pred, dim=1)
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
    
    train_files =  get_file_names('train.txt')
    test_files = get_file_names('test.txt')
    val_files = get_file_names('val.txt')

    training_data_size = 3
    testing_data_size =  2
    validation_data_size = 3
    train_feature_file_name = f"{training_data_size}_resnet50_train_features.pt" # 30_ for 30 videos
    test_feature_file_name =  f"{testing_data_size}_resnest50_test_features.pt"
    training_frame_range = (0,5)
    testing_frame_range = (5,8)

    train_files = train_files[0:training_data_size]
    val_files = val_files[0:validation_data_size]

    #print("count of total unique labels is", len(get_unique_labels(train_files, val_files)))

    if extract_features:
        print("Extractingg")
        saving_time_start = time.time()
        train_dataset = VODDataset(video_names=train_files, split="train", frames=training_frame_range)
        test_dataset = VODDataset(video_names=val_files, split="test", frames=testing_frame_range)
        
        train_loader = DataLoader(train_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=False)
        save_features(train_loader, train_feature_file_name, aggregate=True)
        save_features(test_loader, test_feature_file_name, aggregate=True)
        saving_time_end = time.time()
        print(f"Time taken to save train and test features is {saving_time_end-saving_time_start}")
        
    #After features are saved, load and train + test data
    train_video_features = torch.load(train_feature_file_name)
    features = train_video_features["features"]
    # print("features shape", features.shape)
    labels = train_video_features["labels"]
    # print("labels shape", labels.shape)
    new_train_dataset = TensorDataset(features, labels)
    new_train_loader = DataLoader(new_train_dataset, batch_size=32, shuffle=True)
    
    test_video_features = torch.load(test_feature_file_name)
    features = test_video_features["features"] # 9,2048
    labels = test_video_features["labels"] # 9,10
    new_test_dataset = TensorDataset(features, labels)
    new_test_loader = DataLoader(new_test_dataset, batch_size=16, shuffle=False)
    
    model = CNN_Classifier(num_classes=len(get_unique_labels(train_files, val_files))).to(device) # TODO: fix num_classes, get unique object/action labels based on videos
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