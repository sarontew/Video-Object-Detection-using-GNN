"""
CNN Baseline v1 — feature-based classifier with dynamic fold val
CNN Baseline v2 — end-to-end backbone classifier with dynamic fold val

Both scripts:
  - Load the same splits JSON as the GNN script (fixed train/test)
  - Carve out val dynamically via StratifiedGroupKFold (same K and seed)
  - Use identical EarlyStopping constants
  - Evaluate the globally best checkpoint on the fixed test set
"""

# ══════════════════════════════════════════════════════════════════════════════
# RUN CNN BASELINE V1:
#   python cnn_baselines.py --version 1
# RUN CNN BASELINE V2:
#   python cnn_baselines.py --version 2 [--ExtractFeatures 1]
# ══════════════════════════════════════════════════════════════════════════════

import os
import copy
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from torch.utils.data import (Dataset, TensorDataset, DataLoader, Subset)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import VODDataset, OtherVODDataset
from cnn_baseline import CNN_Classifier, CNN_With_Backbone
from utils import get_file_names, save_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              SHARED CONFIGURATION — mirrors GNN script                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MIN_SAMPLES          = 10
OUTPUT_FILE          = f"splits_{MIN_SAMPLES}.json"

# Must match GNN script exactly for a fair comparison
EARLY_STOP_PATIENCE  = 5
EARLY_STOP_MIN_DELTA = 0.03
NUM_EPOCHS           = 20
BATCH_SIZE           = 16

# Per-version learning rates (set in __main__)
LR_V1 = 1e-4
LR_V2 = 1e-3

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_label_from_filename(filename):
    name  = filename.split('.')[0]
    parts = name.split('_')
    return "_".join(parts[2:])

def get_object_label(video_name):
    parts = video_name.split('.')[0].split('_')
    return "_".join(parts[2:])

# ── Early stopping — identical to GNN script ──────────────────────────────────

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float('inf')
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  Early stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True

    def reset(self):
        self.best_loss   = float('inf')
        self.counter     = 0
        self.should_stop = False

# ── Training / evaluation helpers ─────────────────────────────────────────────

def train_one_epoch_v1(loader, model, optimiser, lossfn):
    """CNN v1: loader yields (img, target)"""
    model.train()
    total_loss = 0.0
    for img, target in loader:
        img, target = img.to(device), target.to(device)
        optimiser.zero_grad()
        loss = lossfn(model(img), target)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train_one_epoch_v2(loader, model, optimiser, lossfn):
    """CNN v2: loader yields (img, target, video_id)"""
    model.train()
    total_loss = 0.0
    for img, target, _ in loader:
        img, target = img.to(device), target.to(device)
        optimiser.zero_grad()
        loss = lossfn(model(img), target)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate_v1(model, loader, lossfn):
    """CNN v1: (img, target) batches → (val_loss, val_acc)"""
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for img, target in loader:
            img, target  = img.to(device), target.to(device)
            pred          = model(img)
            total_loss   += lossfn(pred, target).item()
            _, predicted  = torch.max(pred, 1)
            total        += target.size(0)
            correct      += (predicted == target).sum().item()
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def evaluate_v2(model, loader, lossfn):
    """CNN v2: (img, target, video_id) batches → (val_loss, val_acc)"""
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for img, target, _ in loader:
            img, target  = img.to(device), target.to(device)
            pred          = model(img)
            total_loss   += lossfn(pred, target).item()
            _, predicted  = torch.max(pred, 1)
            total        += target.size(0)
            correct      += (predicted == target).sum().item()
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def test_model(model, loader, lossfn, unique_label_mappings, has_video_id, version):
    """Full test: confusion matrix + frame + video-level accuracy."""
    model.eval()
    test_loss        = 0
    correct_frames   = 0
    video_probs_dict = {}
    all_preds        = []
    all_targets      = []

    with torch.no_grad():
        for batch in loader:
            if has_video_id:
                img, target, _ = batch
            else:
                img, target    = batch
            img, target = img.to(device), target.to(device)
            pred         = model(img)
            test_loss   += lossfn(pred, target).item()

            _, predicted = torch.max(pred, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            correct_frames += (pred.argmax(1) == target).type(torch.float).sum().item()

            probs = F.softmax(pred, dim=1)
            for i, vid in enumerate(target):
                vid = vid.item()
                if vid not in video_probs_dict:
                    video_probs_dict[vid] = []
                video_probs_dict[vid].append(probs[i].cpu())

    cm            = confusion_matrix(all_targets, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    cm_norm       = cm.astype("float") / cm.sum(axis=1)[:, None]

    print("\nConfusion Matrix:"); print(cm)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Class {i}: {acc*100:.2f}%")

    class_names = list(unique_label_mappings.keys())
    plt.figure(figsize=(max(6, len(class_names)//2), max(5, len(class_names)//2)))
    sns.heatmap(cm_norm, xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", fmt=".2f")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Normalised Confusion Matrix (CNN v{version})")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_v{version}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds))

    correct_videos = 0
    for vid, prob_list in video_probs_dict.items():
        pred_class = torch.stack(prob_list).mean(dim=0).argmax().item()
        if pred_class == vid:
            correct_videos += 1

    frame_acc  = correct_frames / max(len(loader.dataset), 1)
    video_acc  = correct_videos / max(len(video_probs_dict), 1)
    test_loss /= max(len(loader), 1)
    print(f"Frame Accuracy: {100*frame_acc:.1f}%  "
          f"Video Accuracy: {100*video_acc:.1f}%  "
          f"Loss: {test_loss:.6f}")
    return video_acc

# ── Transforms for v2 ─────────────────────────────────────────────────────────

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Feature extraction helper (V1 only) ──────────────────────────────────────

def save_features_for_split(files, unique_label_mappings, split, file_name,
                             chunk_size=50):
    """
    Iterates over `files` in chunks, builds a VODDataset for each chunk,
    and calls save_features() to write ResNet50 features to disk.

    Parameters
    ----------
    files                 : list of video name strings
    unique_label_mappings : dict label_str → class_int
    split                 : 'train' or 'test'  (controls frame range in VODDataset)
    file_name             : base filename passed to save_features()
    chunk_size            : number of videos processed per VODDataset instance
    """
    is_train = (split == 'train')
    for i in range(0, len(files), chunk_size):
        chunk   = files[i:i + chunk_size]
        dataset = VODDataset(video_names=chunk,
                             unique_label_mappings=unique_label_mappings,
                             split=split)
        if len(dataset) == 0:
            print(f"  [save_features_for_split] Empty dataset for chunk {i}—skipping")
            continue
        loader = DataLoader(dataset, batch_size=64, shuffle=is_train)
        save_features(loader, file_name, train=is_train, aggregate=False)
        print(f"  Saved features chunk {i}–{i+len(chunk)} ({split})")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",        type=int, required=True,
                        help="1 = feature-based CNN, 2 = end-to-end CNN")
    parser.add_argument("--ExtractFeatures", type=str, required=False)
    args = parser.parse_args()
    version          = args.version
    extract_features = bool(args.ExtractFeatures)

    assert version in (1, 2), "--version must be 1 or 2"

    # ── Load splits ───────────────────────────────────────────────────────────
    with open(OUTPUT_FILE) as fp:
        splits = json.load(fp)

    train_files           = splits["train"]
    test_files            = splits["test"]
    unique_label_mappings = splits["label_mapping"]
    num_classes           = splits["num_classes"]
    K_FOLDS               = splits["k_folds"]    # identical to GNN
    SEED                  = splits["seed"]

    assert splits.get("val", []) == [], \
        "val should be empty — fold-based val is generated here dynamically"

    print(f"\nCNN Baseline v{version}")
    print(f"Train: {len(train_files)}  Test: {len(test_files)}")
    print(f"Classes: {num_classes}  K_FOLDS: {K_FOLDS}  Seed: {SEED}\n")

    # ── StratifiedGroupKFold — same parameters as GNN ─────────────────────────
    fold_labels = np.array([unique_label_mappings[extract_label_from_filename(f)]
                             for f in train_files])
    fold_groups = np.array(train_files)
    sgkf        = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True,
                                       random_state=SEED)

    # ── Global best tracking ───────────────────────────────────────────────────
    global_best_val_loss  = float('inf')
    global_best_state     = None
    fold_summaries        = []
    training_start        = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    #  V1: FEATURE-BASED (pre-extracted ResNet50 features)
    # ══════════════════════════════════════════════════════════════════════════
    if version == 1:
        LR = LR_V1

        # Optional feature extraction
        if extract_features:
            save_features_for_split(train_files, unique_label_mappings,
                                    'train', "resnet50_train_features.pt")
            save_features_for_split(test_files,  unique_label_mappings,
                                    'test',  "resnest50_test_features.pt")

        # Load all pre-extracted train features
        all_train_features, all_train_labels, video_ids = [], [], []
        for fname in sorted(os.listdir("Resnet50_Features/train")):
            frame = torch.load(f"Resnet50_Features/train/{fname}")
            all_train_features.append(frame['features'].unsqueeze(0))
            all_train_labels.append(torch.tensor(frame['labels']))
            video_ids.append(fname)

        features  = torch.cat(all_train_features, dim=0)
        labels    = torch.stack(all_train_labels)
        labels_np = labels.cpu().numpy()
        groups    = np.array(video_ids)

        full_train_dataset = TensorDataset(features, labels)

        # Override fold_labels/groups with actual feature file labels
        # (file ordering may differ from train_files list)
        fold_labels_v1 = labels_np
        fold_groups_v1 = groups
        sgkf_v1        = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True,
                                               random_state=SEED)

        for fold_idx, (tr_idx, val_idx) in enumerate(
            sgkf_v1.split(features, fold_labels_v1, fold_groups_v1)
        ):
            print(f"\n{'='*55} Fold {fold_idx+1}/{K_FOLDS} {'='*5}")
            print(f"  mini-train={len(tr_idx)}  val={len(val_idx)}")

            train_loader = DataLoader(
                full_train_dataset, batch_size=BATCH_SIZE,
                sampler=torch.utils.data.SubsetRandomSampler(tr_idx))
            val_loader   = DataLoader(
                full_train_dataset, batch_size=BATCH_SIZE,
                sampler=torch.utils.data.SubsetRandomSampler(val_idx))

            model     = CNN_Classifier(num_classes=num_classes).to(device)
            optimiser = torch.optim.Adam(model.parameters(), lr=LR)
            lossfn    = nn.CrossEntropyLoss().to(device)

            early_stopper    = EarlyStopping(EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA)
            best_fold_loss   = float('inf')
            best_fold_state  = None
            stopped_at_epoch = NUM_EPOCHS

            for epoch in range(NUM_EPOCHS):
                train_loss           = train_one_epoch_v1(train_loader, model,
                                                          optimiser, lossfn)
                val_loss, val_acc    = evaluate_v1(model, val_loader, lossfn)

                print(f"  Epoch {epoch:02d}  train={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={100*val_acc:.1f}%")

                if val_loss < best_fold_loss:
                    best_fold_loss  = val_loss
                    best_fold_state = {k: v.clone()
                                       for k, v in model.state_dict().items()}
                    print(f"  → Checkpoint (val_loss {best_fold_loss:.4f})")

                early_stopper.step(val_loss)
                if early_stopper.should_stop:
                    print(f"  Early stopping at epoch {epoch}")
                    stopped_at_epoch = epoch + 1
                    break

            if best_fold_loss < global_best_val_loss:
                global_best_val_loss = best_fold_loss
                global_best_state    = best_fold_state
                print(f"  ★ New global best (val loss {global_best_val_loss:.4f})")

            fold_summaries.append({
                "fold": fold_idx+1, "best_val_loss": round(best_fold_loss, 4),
                "stopped_epoch": stopped_at_epoch, "val_acc": round(val_acc, 4),
            })

        # Load pre-extracted test features
        all_test_features, all_test_labels = [], []
        for fname in sorted(os.listdir("Resnet50_Features/test")):
            frame = torch.load(f"Resnet50_Features/test/{fname}")
            all_test_features.append(frame['features'].unsqueeze(0))
            all_test_labels.append(torch.tensor(frame['labels']))
        test_feats   = torch.cat(all_test_features, dim=0)
        test_lbls    = torch.stack(all_test_labels)
        test_dataset = TensorDataset(test_feats, test_lbls)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        final_model = CNN_Classifier(num_classes=num_classes).to(device)
        final_model.load_state_dict(global_best_state)

        print("\n" + "="*60 + "\nFINAL TEST EVALUATION (CNN v1)\n" + "="*60)
        test_model(final_model, test_loader, lossfn,
                   unique_label_mappings, has_video_id=False, version=1)
        torch.save(global_best_state, 'cnn_baseline_v1_best.pt')

    # ══════════════════════════════════════════════════════════════════════════
    #  V2: END-TO-END BACKBONE (OtherVODDataset, raw frames)
    # ══════════════════════════════════════════════════════════════════════════
    elif version == 2:
        LR = LR_V2

        full_dataset = OtherVODDataset(train_files, unique_label_mappings,
                                        split='train')
        test_dataset = OtherVODDataset(test_files,  unique_label_mappings,
                                        split='test')

        # Build per-frame labels and groups for StratifiedGroupKFold
        frame_labels, frame_groups = [], []
        for i in range(len(full_dataset)):
            _, lbl, vid_id = full_dataset[i]
            frame_labels.append(lbl.item())
            frame_groups.append(vid_id)

        frame_labels_np = np.array(frame_labels)
        frame_groups_np = np.array(frame_groups)

        for fold_idx, (tr_idx, val_idx) in enumerate(
            sgkf.split(np.zeros(len(frame_labels_np)),
                        frame_labels_np, frame_groups_np)
        ):
            print(f"\n{'='*55} Fold {fold_idx+1}/{K_FOLDS} {'='*5}")
            print(f"  mini-train frames={len(tr_idx)}  val frames={len(val_idx)}")

            # Deep-copy so each split gets its own transform
            train_ds = copy.deepcopy(full_dataset)
            val_ds   = copy.deepcopy(full_dataset)
            train_ds.transform = train_transform
            val_ds.transform   = val_transform

            train_loader = DataLoader(Subset(train_ds, tr_idx),
                                      batch_size=BATCH_SIZE, shuffle=True,
                                      drop_last=True)
            val_loader   = DataLoader(Subset(val_ds, val_idx),
                                      batch_size=BATCH_SIZE, shuffle=False)

            model     = CNN_With_Backbone(num_classes=num_classes).to(device)
            optimiser = torch.optim.Adam(model.parameters(), lr=LR)
            lossfn    = nn.CrossEntropyLoss().to(device)

            early_stopper    = EarlyStopping(EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA)
            best_fold_loss   = float('inf')
            best_fold_state  = None
            stopped_at_epoch = NUM_EPOCHS

            for epoch in range(NUM_EPOCHS):
                train_loss        = train_one_epoch_v2(train_loader, model,
                                                       optimiser, lossfn)
                val_loss, val_acc = evaluate_v2(model, val_loader, lossfn)

                print(f"  Epoch {epoch:02d}  train={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={100*val_acc:.1f}%")

                if val_loss < best_fold_loss:
                    best_fold_loss  = val_loss
                    best_fold_state = {k: v.clone()
                                       for k, v in model.state_dict().items()}
                    print(f"  → Checkpoint (val_loss {best_fold_loss:.4f})")

                early_stopper.step(val_loss)
                if early_stopper.should_stop:
                    print(f"  Early stopping at epoch {epoch}")
                    stopped_at_epoch = epoch + 1
                    break

            if best_fold_loss < global_best_val_loss:
                global_best_val_loss = best_fold_loss
                global_best_state    = best_fold_state
                print(f"  ★ New global best (val loss {global_best_val_loss:.4f})")

            fold_summaries.append({
                "fold": fold_idx+1, "best_val_loss": round(best_fold_loss, 4),
                "stopped_epoch": stopped_at_epoch, "val_acc": round(val_acc, 4),
            })

        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        final_model = CNN_With_Backbone(num_classes=num_classes).to(device)
        final_model.load_state_dict(global_best_state)

        print("\n" + "="*60 + "\nFINAL TEST EVALUATION (CNN v2)\n" + "="*60)
        test_model(final_model, test_loader, lossfn,
                   unique_label_mappings, has_video_id=True, version=2)
        torch.save(global_best_state, 'cnn_baseline_v2_best.pt')

    # ── Shared summary ─────────────────────────────────────────────────────────
    training_end = time.time()
    print(f"\nTraining time: {(training_end - training_start)/60:.1f}min")
    print("\nFold summary:")
    for s in fold_summaries:
        print(f"  Fold {s['fold']}: val_loss={s['best_val_loss']:.4f}  "
              f"val_acc={100*s['val_acc']:.1f}%  "
              f"stopped@ep{s['stopped_epoch']}")