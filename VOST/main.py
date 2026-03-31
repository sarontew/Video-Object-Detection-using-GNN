import os
import cv2
import time
import torch
import random
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
from sklearn.model_selection import KFold, StratifiedGroupKFold
from utils import get_file_names, get_unique_labels, save_features
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save(files, chunck_size, unique_label_mappings, split, file_name ):
    print("split", split)
    print("filename", file_name)
    for i in range(0, len(files), chunck_size):
        print("Extractingg")
        ffs = files[i:i+chunck_size]
        dataset = VODDataset(video_names=ffs, unique_label_mappings= unique_label_mappings, split=split)
        if len(dataset) == 0:
            print("empty dataset?")
        else:
            loader = DataLoader(
                dataset,
                batch_size=64,
                shuffle=True
            )
            train = True if split == "train" else False
            save_features(loader, file_name, train=train, aggregate=False)

def get_object_label(video_name):
    parts = video_name.split('.')[0].split('_')
    return "_".join(parts[2:])

## TRAIN
def train(train_loader, model, optimiser, lossfn, fold):
    #for epoch in range(20):
    model.train()
    running_loss = 0.0
    for img, target in train_loader:
        images = img.to(device) # Video feature input with dim 1x2028
        target = target.to(device)
        optimiser.zero_grad()
        # print("images type is ", type(images))
        # print("cnn images input dim is", images.shape)
        pred = model(images)
        # print("pred is", pred)
        # print("target is", target)
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
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for img, target in test_loader:
            img = img.to(device)
            target = target.to(device)
            
            pred = model(img)  # (batch_size, num_classes)
            # print("pred argmax", pred.argmax(1))
            # print("target", target)
            test_loss += lossfn(pred, target).item()

            # predicted class
            _, predicted = torch.max(pred, 1)

            # for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
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

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)

    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"Class {i}: {acc*100:.2f}%")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalised Confusion Matrix")
    plt.show()

    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


    plt.figure(figsize=(6,5))
    class_names = list(unique_label_mappings.keys())
    sns.heatmap(cm_norm, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalised Confusion Matrix with class names")
    plt.show()

    plt.savefig("confusion_matrix_with_labels.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))

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

# def extract_label_from_filename(filename):
#     # Split by underscore, take last part before extension
#     return filename.split('_')[-1].split('.')[0]  # 'cloth' in your example

def extract_label_from_filename(filename):
    name = filename.split('.')[0]        # remove extension if present
    parts = name.split('_')
    return "_".join(parts[2:])           # join everything after action

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument("--ExtractFeatures", type=str,
                        required=False)
    args = parser.parse_args()
    extract_features = bool(args.ExtractFeatures)

    all_valid_file_names = get_file_names('train.txt') + get_file_names('val.txt')

    from collections import defaultdict
    import random

    MIN_SAMPLES = 3
    MAX_SAMPLES = 50
    TEST_SPLIT = 0.2

    class_to_files = defaultdict(list)

    # Build mapping
    for file in all_valid_file_names:
        label = get_object_label(file)
        class_to_files[label].append(file)

    print("Before filtering:", len(class_to_files), "classes")

    # remove small classes
    filtered_class_to_files = {}
    for cls, files in class_to_files.items():
        if len(files) >= MIN_SAMPLES:
            filtered_class_to_files[cls] = files

    print("After filtering:", len(filtered_class_to_files), "classes")

    # cap large classes (optional but recommended)
    for cls, files in filtered_class_to_files.items():
        random.shuffle(files)
        if len(files) > MAX_SAMPLES:
            filtered_class_to_files[cls] = files[:MAX_SAMPLES]

    # Continue with filtered data
    all_classes = list(filtered_class_to_files.keys())
    random.shuffle(all_classes)

    overlap_fraction = 1.0
    num_overlap_classes = max(1, int(len(all_classes) * overlap_fraction))
    overlap_classes = set(all_classes[:num_overlap_classes])

    print("Overlap classes:", overlap_classes)

    train_files = []
    test_files = []

    for cls, files in filtered_class_to_files.items():
        random.shuffle(files)

        split_idx = int((1 - TEST_SPLIT) * len(files))
        
        train_files.extend(files[:split_idx])
        test_files.extend(files[split_idx:])

    all_files = train_files + test_files

    print("Train size:", len(train_files))
    print("Test size:", len(test_files))

    # Extract all unique object names
    all_labels = sorted({extract_label_from_filename(f) for f in all_files})
    #all_valid_file_names = all_valid_file_names[0:2]

    #all_valid_file_names = ['4176_cut_cloth', '4174_cut_cloth', '4331_cut_cloth', '4320_tear_dough', '226_squeeze_dough', '2218_empty_raisin', '455_fold_box', '1186_cut_chilli', '1184_cut_chilli']

    # class_to_files = defaultdict(list)

    # for file in all_valid_file_names:
    #     label = get_object_label(file)
    #     class_to_files[label].append(file)
    
    # all_classes = list(class_to_files.keys())
    # random.shuffle(all_classes)
    # overlap_fraction = 1 # was 0.18
    # num_overlap_classes = max(1, int(len(all_classes) * overlap_fraction))
    # overlap_classes = set(all_classes[:num_overlap_classes])
    # print("overlap classes are", overlap_classes)
    # train_files = []
    # test_files = []
    # for cls, files in class_to_files.items():
    #     random.shuffle(files)
    #     if cls in overlap_classes:
    #         split_idx = int(0.8 * len(files))
    #         train_files.extend(files[:split_idx])
    #         test_files.extend(files[split_idx:])
    #     else:
    #         # Assign whole class to train (or randomly choose)
    #         train_files.extend(files)

    # # training_data_size = int(0.8 * len(all_valid_file_names))
    # # train_files = all_valid_file_names[0:training_data_size]
    # # test_files = all_valid_file_names[training_data_size:]

    # # print("train files", train_files)
    # # print("test files", test_files)

    train_feature_file_name = f"resnet50_train_features.pt" # 30_ for 30 videos
    test_feature_file_name =  f"resnest50_test_features.pt"
    # training_frame_range = (0,5)
    # testing_frame_range = (5,8)

    # all_files = train_files + test_files

    # # Extract all unique object names
    # all_labels = sorted({extract_label_from_filename(f) for f in all_files})
    #print("all labels", all_labels)

    # Create consistent label → ID mapping
    unique_label_mappings = {label: idx for idx, label in enumerate(all_labels)}
    num_classes = len(unique_label_mappings)

    if extract_features:
        save(train_files, 50, unique_label_mappings, 'train', train_feature_file_name )
        save(test_files, 50, unique_label_mappings, 'test', test_feature_file_name )

    ## organise feats after saving
      
    all_train_features = []
    all_train_labels = []
    grouping_video_ids = []

    for train_feats_file_name in os.listdir("Resnet50_Features/train"):
        train_frame = torch.load(f"Resnet50_Features/train/{train_feats_file_name}")
        feat = train_frame['features'].unsqueeze(0)  # shape becomes [1, 2048]
        # print("feat shape", feat.shape)
        all_train_features.append(feat)
        all_train_labels.append(torch.tensor(train_frame['labels']))
        video_id = train_feats_file_name
        # video_id = train_feats_file_name[5]  # adjust if needed
        grouping_video_ids.append(video_id)

    features = torch.cat(all_train_features, dim=0)
    labels = torch.stack(all_train_labels)

    # for stratified k fold
    labels_np = labels.cpu().numpy()
    groups = np.array(grouping_video_ids)

    new_train_dataset = TensorDataset(features, labels)

    # KFold validation
    k_folds = 5
    num_epochs = 10
    batch_size = 2

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    strkfold  = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_val_acc = 0
    best_model_state = None

    print("in main labels np", labels_np)
    print("groups", groups)

    for fold, (train_ids, val_ids) in enumerate(strkfold.split(features, labels_np, groups)):
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

        print("num classes for training and validation is", num_classes)

        model = CNN_Classifier(
            num_classes=num_classes # does it need test files also as possible classes?
        ).to(device)

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
        lossfn = nn.CrossEntropyLoss().to(device)

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
        all_test_features.append(test_frame['features'].unsqueeze(0) )
        all_test_labels.append(torch.tensor(test_frame['labels']))

    features = torch.cat(all_test_features, dim=0)
    labels = torch.stack(all_test_labels)

    # print("features dim", features.shape)
    # print("labels dim", labels.shape)
    # print("Test labels unique:", torch.unique(labels))
    # print("Num classes:", num_classes)

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

    torch.save(model.state_dict(), 'cnn_baseline_1.0')