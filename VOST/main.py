import os
import cv2
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torchvision.models.detection import FasterRCNN
from torchvision.models.feature_extraction import get_graph_node_names
from dataset import VODDataset
from torch.utils.data import Dataset, TensorDataset, DataLoader
from cnn_baseline import CNN_Classifier
from sklearn.model_selection import KFold
from utils import get_file_names, get_unique_labels, save_features

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


## TRAIN
def train(train_loader, model, optimiser, lossfn, fold):
    #for epoch in range(20):
    model.train()
    running_loss = 0.0
    for img, target in train_loader:
        images = img.to(device) # Video feature input with dim 1x2028
        optimiser.zero_grad()
        pred = model(images)
        loss = lossfn(pred, target)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()

        # print(f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}")

    print("final loss is ",  loss)

## TEST
def test(model, test_loader, lossfn, device):
    """ Calculates the video accuracy based on highest softmax probability """
    model.eval()
    test_loss = 0
    correct_frames = 0  # frame-level accuracy
    video_probs_dict = {}  # accumulate frame probs per video

    with torch.no_grad():
        for img, target in test_loader:
            img = img.to(device)
            target = target.to(device)
            
            pred = model(img)  # (batch_size, num_classes)
            test_loss += lossfn(pred, target).item()
            
            # frame-level accuracy
            correct_frames += (pred.argmax(1) == target).type(torch.float).sum().item()
            
            # softmax probabilities
            probs = F.softmax(pred, dim=1)
            
            # accumulate per video (using target as video ID)
            for i, vid in enumerate(target):
                vid = vid.item()
                if vid not in video_probs_dict:
                    video_probs_dict[vid] = []
                video_probs_dict[vid].append(probs[i].cpu())

    # compute video-level predictions
    correct_videos = 0
    for vid, prob_list in video_probs_dict.items():
        avg_prob = torch.stack(prob_list).mean(dim=0)
        pred_class = avg_prob.argmax().item()
        if pred_class == vid:  # video-level prediction matches target
            correct_videos += 1

    size = len(test_loader.dataset)
    num_videos = len(video_probs_dict)
    frame_acc = correct_frames / size
    video_acc = correct_videos / num_videos
    test_loss /= len(test_loader)

    print(f"Frame Accuracy: {100*frame_acc:.1f}%, Video Accuracy: {100*video_acc:.1f}%, Loss: {test_loss:.6f}")

def extract_label_from_filename(filename):
    # Split by underscore, take last part before extension
    return filename.split('_')[-1].split('.')[0]  # 'cloth' in your example

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--ExtractFeatures", type=str,
                        required=False)
    args = parser.parse_args()
    extract_features = bool(args.ExtractFeatures)

    all_valid_file_names = get_file_names('train.txt') + get_file_names('val.txt')
    all_valid_file_names = all_valid_file_names[0:20]

    training_data_size = int(0.8 * len(all_valid_file_names))
    train_files = all_valid_file_names[0:training_data_size]
    test_files = all_valid_file_names[training_data_size:]

    train_feature_file_name = f"resnet50_train_features.pt" # 30_ for 30 videos
    test_feature_file_name =  f"resnest50_test_features.pt"
    training_frame_range = (0,5)
    testing_frame_range = (5,8)

    all_files = train_files + test_files

    # Extract all unique object names
    all_labels = sorted({extract_label_from_filename(f) for f in all_files})

    # Create consistent label → ID mapping
    unique_label_mappings = {label: idx for idx, label in enumerate(all_labels)}
    num_classes = len(unique_label_mappings)
    print("Classes:", unique_label_mappings)
    print("Total classes:", num_classes)

    if extract_features:
        print("Extractingg")
        saving_time_start = time.time()
        train_dataset = VODDataset(video_names=train_files, unique_label_mappings= unique_label_mappings, split="train")
        test_dataset = VODDataset(video_names=test_files, unique_label_mappings= unique_label_mappings, split="test")
        
        train_loader = DataLoader(train_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=False)

        save_features(train_loader, train_feature_file_name, train=True, aggregate=False)
        save_features(test_loader, test_feature_file_name, train=False, aggregate=False)
        saving_time_end = time.time()
        print(f"Time taken to save train and test features is {saving_time_end-saving_time_start}")
        
    all_train_features = []
    all_train_labels = []

    for train_feats_file_name in os.listdir("Resnet50_Features/train"):
        train_frame = torch.load(f"Resnet50_Features/train/{train_feats_file_name}")
        all_train_features.append(train_frame['features'])
        all_train_labels.append(torch.tensor(train_frame['labels']))
 
    features = torch.cat(all_train_features, dim=0)
    labels = torch.stack(all_train_labels)
    new_train_dataset = TensorDataset(features, labels)

    # KFold validation
    k_folds = 5
    num_epochs = 10
    batch_size = 2

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_val_acc = 0
    best_model_state = None

    for fold, (train_ids, val_ids) in enumerate(kfold.split(new_train_dataset)):
        print(f"\n========== Fold {fold+1}/{k_folds} ==========")

        # Sample elements randomly from a given list of ids
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and validation
        train_loader = DataLoader(
            new_train_dataset,
            batch_size=batch_size,
            sampler=train_subsampler
        )

        val_loader = DataLoader(
            new_train_dataset,
            batch_size=batch_size,
            sampler=val_subsampler
        )

        model = CNN_Classifier(
            num_classes=num_classes # does it need test files also as possible classes?
        ).to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
        lossfn = nn.CrossEntropyLoss()

        train(train_loader, model, optimiser, lossfn, fold)

        # Validation step
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    all_test_features = []
    all_test_labels = []

    for test_feats_file_name in os.listdir("Resnet50_Features/test"):
        test_frame = torch.load(f"Resnet50_Features/test/{test_feats_file_name}")
        all_test_features.append(test_frame['features'])
        all_test_labels.append(torch.tensor(test_frame['labels']))

    features = torch.cat(all_test_features, dim=0)
    labels = torch.stack(all_test_labels)

    new_test_dataset = TensorDataset(features, labels)
    new_test_loader = DataLoader(new_test_dataset, batch_size=16, shuffle=False)

    training_start_time = time.time()
    training_end_time = time.time()

    final_model = CNN_Classifier(
        num_classes=num_classes
    ).to(device)
    final_model.load_state_dict(best_model_state) # best validation states
    
    testing_start_time = time.time()
    test(final_model, new_test_loader, lossfn, device)
    testing_end_time = time.time()
    
    train_time = training_end_time - training_start_time
    test_time = testing_end_time - testing_start_time
    print(f"Train time is {train_time} and test time is {test_time}")