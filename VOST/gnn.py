"""
CNN + GNN training script with ablation toggles

Changes over previous version:
  1. num_workers / pin_memory / persistent_workers / prefetch_factor on all loaders
  2. Class weights REMOVED from both loss functions
  3. CNN frame-level accuracy tracked and printed separately from video-level
  4. CNN backbone t-SNE collected at epoch 0 (before GNN gradients) AND at
     the final epoch (after), plotted side-by-side for before/after comparison
"""

import os, json, time, math, random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, pairwise_distances, silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch.utils.data import DataLoader as TorchDataLoader
from skimage.measure import label, regionprops

from cnn_baseline import FrameClassifierUnifiedCNN
from gnn_dataset import New_GNN_Dataset
from utils import get_object_label, extract_label_from_filename, get_file_names

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       SESSION CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SESSION_NAME  = "s0-death-matched-gnn-dim-multi-workers-minarea30"

# ── Ablation toggles ────────────────────────────────────────────────────────
USE_AUGMENTATION = False
K_FOLDS          = 1

SESSION_NOTES = (
    f"aug={'on' if USE_AUGMENTATION else 'off'}  folds={K_FOLDS}  "
    "no class weights, frame-level CNN acc, before/after CNN t-SNE"
)

# ── Model / training hyperparameters ───────────────────────────────────────
GNN_LR              = 1e-3
CNN_LR              = 5e-4
NUM_EPOCHS          = 30
CHUNK_SIZE          = 30
GNN_LR_STEP         = 10
GNN_LR_GAMMA        = 0.5
GNN_WEIGHT_DECAY    = 1e-3
CNN_WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING     = 0.1
GNN_DROPOUT         = 0.3
GNN_HIDDEN_DIM      = 2048
INTRA_FRAME_EDGES   = True
MIN_REGION_AREA     = 30
LAMBDA              = 0.5

EARLY_STOP_PATIENCE  = 5
EARLY_STOP_MIN_DELTA = 0.02

SYNTHETIC_MODE        = False
SYNTHETIC_NOISE_SCALE = 0.1
SYNTHETIC_FEAT_DIM    = 2048
MAX_PER_CLASS_TSNE    = 70

# ── Runtime ──────────────────────────────────────────────────────────────────
NUM_WORKERS     = 4   # parallel frame/mask I/O workers
PREFETCH_FACTOR = 2   # batches prefetched per worker

# ── Output directory ─────────────────────────────────────────────────────────
aug_tag = "aug" if USE_AUGMENTATION else "noaug"
OUT_DIR = f"outputs/{SESSION_NAME}-{aug_tag}-{K_FOLDS}fold"
os.makedirs(OUT_DIR, exist_ok=True)

def spath(filename):
    return os.path.join(OUT_DIR, filename)

def make_loader(dataset, shuffle):
    """Shared DataLoader factory with worker / prefetch settings."""
    return TorchDataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )

print("\n" + "█" * 65)
print(f"  SESSION : {SESSION_NAME}")
print(f"  NOTES   : {SESSION_NOTES}")
print(f"  USE_AUGMENTATION={USE_AUGMENTATION}  K_FOLDS={K_FOLDS}")
print(f"  GNN LR={GNN_LR}  CNN LR={CNN_LR}  Epochs/fold={NUM_EPOCHS}")
print(f"  NUM_WORKERS={NUM_WORKERS}  PREFETCH={PREFETCH_FACTOR}")
print(f"  Outputs → {OUT_DIR}/")
print("█" * 65 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA']   = "1"

MIN_SAMPLES = 10
OUTPUT_FILE = f"splits_{MIN_SAMPLES}.json"

with open(OUTPUT_FILE) as fp:
    splits = json.load(fp)

train_files           = splits["train"]
test_files            = splits["test"]
unique_label_mappings = splits["label_mapping"]
num_classes           = splits["num_classes"]
SEED                  = splits["seed"]
assert splits.get("val", []) == [], \
    "val should be empty — generated dynamically here"

print(f"Train: {len(train_files)}  Test: {len(test_files)}")
print(f"Classes: {num_classes}  K_FOLDS: {K_FOLDS}  Seed: {SEED}\n")

with open(spath("config.json"), "w") as fp:
    json.dump(dict(
        session_name=SESSION_NAME, session_notes=SESSION_NOTES,
        use_augmentation=USE_AUGMENTATION, k_folds=K_FOLDS,
        gnn_lr=GNN_LR, cnn_lr=CNN_LR, num_epochs=NUM_EPOCHS,
        gnn_weight_decay=GNN_WEIGHT_DECAY, cnn_weight_decay=CNN_WEIGHT_DECAY,
        label_smoothing=LABEL_SMOOTHING, gnn_dropout=GNN_DROPOUT,
        gnn_hidden_dim=GNN_HIDDEN_DIM, early_stop_patience=EARLY_STOP_PATIENCE,
        intra_frame_edges=INTRA_FRAME_EDGES, lambda_=LAMBDA,
        min_region_area=MIN_REGION_AREA, seed=SEED,
        num_workers=NUM_WORKERS,
    ), fp, indent=2)

# ── Build fold splits ─────────────────────────────────────────────────────────
fold_labels = np.array([unique_label_mappings[extract_label_from_filename(f)]
                         for f in train_files])
fold_groups = np.array(train_files)

if K_FOLDS == 1:
    rng = np.random.default_rng(SEED)
    mini_train_idx, val_idx = [], []
    for cls in np.unique(fold_labels):
        cls_idx = np.where(fold_labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * 0.2))
        val_idx.extend(cls_idx[:n_val].tolist())
        mini_train_idx.extend(cls_idx[n_val:].tolist())
    fold_splits = [(np.array(mini_train_idx), np.array(val_idx))]
    print(f"K_FOLDS=1: mini-train={len(mini_train_idx)}  val={len(val_idx)}")
else:
    sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(sgkf.split(np.zeros(len(fold_labels)), fold_labels, fold_groups))
    print(f"K_FOLDS={K_FOLDS}: StratifiedGroupKFold splits generated")

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_sinusoidal_base_vectors(num_classes, d_model):
    pe       = torch.zeros(num_classes, d_model)
    position = torch.arange(num_classes, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
    return pe

def synthetic_node_feature(class_idx, base_vectors, noise_scale, device):
    base = base_vectors[class_idx].clone()
    feat = (base + torch.randn_like(base) * noise_scale).to(device)
    return feat.unsqueeze(-1).unsqueeze(-1)

def mask_to_feat_soft_area(mask_hw, feat):
    _, HF, WF = feat.shape
    m = mask_hw.float()[None, None, ...]
    return F.interpolate(m, size=(HF, WF), mode="area")

def masked_avrg_pool(mask_feat, feat_map):
    return (mask_feat * feat_map).sum(dim=(2, 3)) / mask_feat.sum().clamp_min(1e-6)

def pool_object_vec(feat_map, mask):
    mask      = torch.as_tensor(mask, dtype=torch.float32, device=feat_map.device)
    mask_feat = mask_to_feat_soft_area(mask, feat_map)
    obj_vec   = masked_avrg_pool(mask_feat, feat_map.unsqueeze(0))
    return obj_vec.squeeze(0).unsqueeze(-1).unsqueeze(-1)

def get_class_name(idx):
    return [k for k, v in unique_label_mappings.items() if v == idx][0]

base_vectors = make_sinusoidal_base_vectors(num_classes, SYNTHETIC_FEAT_DIM)

# ── GNN model ─────────────────────────────────────────────────────────────────

class St_GCN_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        H = GNN_HIDDEN_DIM
        self.node_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(SYNTHETIC_FEAT_DIM, H), nn.ReLU(),
            nn.Dropout(p=GNN_DROPOUT),
        )
        self.conv1      = GCNConv(H, H)
        self.drop       = nn.Dropout(p=GNN_DROPOUT)
        self.classifier = nn.Linear(H, num_classes)

    def forward(self, data, return_hidden=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x      = self.node_proj(x)
        x      = x + self.drop(F.relu(self.conv1(x, edge_index)))
        pooled = global_mean_pool(x, batch)
        if return_hidden:
            return self.classifier(pooled), pooled
        return self.classifier(pooled)

# ── Graph building ─────────────────────────────────────────────────────────────

def build_graph(subimages_per_frame, device):
    frame_ids = sorted(subimages_per_frame.keys())
    frame_offsets = {}
    offset = 0
    for fid in frame_ids:
        frame_offsets[fid] = offset
        offset += len(subimages_per_frame[fid])

    all_nodes, edges_from, edges_to = [], [], []
    for idx, fid in enumerate(frame_ids):
        nodes_here = subimages_per_frame[fid]
        all_nodes.extend(nodes_here)
        if INTRA_FRAME_EDGES:
            base = frame_offsets[fid]
            for ci in range(len(nodes_here)):
                for cj in range(len(nodes_here)):
                    if ci != cj:
                        edges_from.append(base + ci)
                        edges_to.append(base + cj)

    if not all_nodes:
        return None

    nodes_tensor = torch.stack(all_nodes)
    edge_index   = torch.tensor([edges_from, edges_to], dtype=torch.long).to(device)
    return Data(x=nodes_tensor, edge_index=edge_index).to(device)


def process_video_to_graph(video_frames, video_masks, video_label,
                            cnn_model, base_vectors, device):
    F_len               = video_frames.shape[0]
    subimages_per_frame = {}
    global_frame_count  = 0
    allcnnlosses        = torch.tensor(0.0, device=device)
    all_logits          = []
    all_feat_vecs       = []
    fallback_count      = 0
    # frame-level: accumulate correct/total across all chunks
    frame_correct       = 0
    frame_total         = 0

    for i in range(0, F_len, CHUNK_SIZE):
        chunk = video_frames[i:i + CHUNK_SIZE].to(device)
        B = chunk.size(0)

        if not SYNTHETIC_MODE:
            logits, feature_maps = cnn_model(chunk)
            all_logits.append(logits)
            feat_vec = feature_maps.mean(dim=(2, 3))
            all_feat_vecs.append(feat_vec.detach())
            video_labels_chunk = video_label.repeat(B)
            # No class weights — plain CrossEntropyLoss
            allcnnlosses = allcnnlosses + cnn_lossfn(logits, video_labels_chunk)
            # Frame-level accuracy
            frame_correct += (logits.argmax(1) == video_labels_chunk).float().sum().item()
            frame_total   += B
        else:
            logits, feature_maps = cnn_model(chunk)
            allcnnlosses = torch.tensor(0.0, requires_grad=True, device=device)

        for j in range(B):
            subimages_per_frame[global_frame_count] = []

            if SYNTHETIC_MODE:
                node_feat = synthetic_node_feature(
                    int(video_label.item()), base_vectors, SYNTHETIC_NOISE_SCALE, device)
                subimages_per_frame[global_frame_count].append(node_feat)
                global_frame_count += 1
            else:
                mask    = video_masks[i + j].float()
                mask_np = mask.cpu().numpy().astype("uint8")
                regions = regionprops(label(mask_np, connectivity=2))
                H, W    = mask_np.shape

                for r in regions:
                    if r.area <= MIN_REGION_AREA:
                        continue
                    rm = torch.zeros((H, W), dtype=torch.float32)
                    rm[r.coords[:, 0], r.coords[:, 1]] = 1.0
                    node_vec = pool_object_vec(feature_maps[j], rm.to(device))
                    subimages_per_frame[global_frame_count].append(node_vec)

                if not subimages_per_frame[global_frame_count]:
                    fallback_count += 1
                    del subimages_per_frame[global_frame_count]
                    continue

                global_frame_count += 1

    if fallback_count > 0:
        print(f"  {fallback_count}/{F_len} frames skipped (no valid regions)")

    graph        = build_graph(subimages_per_frame, device)
    num_chunks   = max(1, F_len // CHUNK_SIZE)
    avg_cnn_loss = allcnnlosses / num_chunks

    backbone_feat_vec = None
    if all_feat_vecs:
        backbone_feat_vec = torch.cat(all_feat_vecs, dim=0).mean(dim=0)

    return graph, avg_cnn_loss, all_logits, backbone_feat_vec, frame_correct, frame_total

# ── Early stopping ─────────────────────────────────────────────────────────────

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
        self.best_loss = float('inf'); self.counter = 0; self.should_stop = False

# ── Epoch runner ───────────────────────────────────────────────────────────────

def run_epoch(loader, train=True,
              tsne_cnn_feats=None, tsne_cnn_lbls=None, cnn_class_counts=None,
              tsne_gnn_feats=None, tsne_gnn_lbls=None, gnn_class_counts=None):
    if train:
        gnn_model.train(); cnn_model.train()
    else:
        gnn_model.eval();  cnn_model.eval()

    total_vids = correct_gnn = correct_cnn_vids = 0
    sum_gnn_loss = sum_cnn_loss = 0.0
    # Frame-level CNN accumulators
    total_frame_correct = 0
    total_frame_count   = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for f, m, l in loader:
            for vid_idx in range(f.size(0)):
                video_frames = f[vid_idx]
                video_label  = l[vid_idx].to(device)
                video_masks  = m[vid_idx]

                if train:
                    gnn_optimiser.zero_grad()
                    cnn_optimiser.zero_grad()

                (graph, avg_cnn_loss, all_logits,
                 backbone_feat, frame_correct, frame_total) = process_video_to_graph(
                    video_frames, video_masks, video_label,
                    cnn_model, base_vectors, device)

                if graph is None:
                    continue

                gnn_pred, hidden = gnn_model(graph, return_hidden=True)
                gnn_loss = gnn_lossfn(gnn_pred, video_label.unsqueeze(0))

                if train:
                    total_loss = (gnn_loss if SYNTHETIC_MODE
                                  else avg_cnn_loss + LAMBDA * gnn_loss)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
                    gnn_optimiser.step()
                    if not SYNTHETIC_MODE:
                        cnn_optimiser.step()
                        cnn_scheduler.step()

                total_vids          += 1
                correct_gnn         += (gnn_pred.argmax(1) == video_label.unsqueeze(0)).float().sum().item()
                sum_gnn_loss        += gnn_loss.item()
                sum_cnn_loss        += avg_cnn_loss.item() if not SYNTHETIC_MODE else 0.0
                total_frame_correct += frame_correct
                total_frame_count   += frame_total

                # Video-level CNN: softmax vote across all frames
                if all_logits:
                    all_probs    = F.softmax(torch.cat(all_logits, dim=0), dim=1)
                    cnn_pred_vid = all_probs.mean(dim=0).argmax().item()
                    correct_cnn_vids += int(cnn_pred_vid == video_label.item())

                lbl_int = int(video_label.item())

                if (tsne_cnn_feats is not None and backbone_feat is not None
                        and cnn_class_counts[lbl_int] < MAX_PER_CLASS_TSNE):
                    tsne_cnn_feats.append(backbone_feat.cpu().numpy().astype(np.float32))
                    tsne_cnn_lbls.append(lbl_int)
                    cnn_class_counts[lbl_int] += 1

                if (tsne_gnn_feats is not None
                        and gnn_class_counts[lbl_int] < MAX_PER_CLASS_TSNE):
                    tsne_gnn_feats.append(
                        hidden.detach().cpu().numpy().astype(np.float32).squeeze(0))
                    tsne_gnn_lbls.append(lbl_int)
                    gnn_class_counts[lbl_int] += 1

    n = max(total_vids, 1)
    return {
        "gnn_acc":       correct_gnn / n,
        "cnn_vid_acc":   correct_cnn_vids / n,
        "cnn_frame_acc": total_frame_correct / max(total_frame_count, 1),
        "gnn_loss":      sum_gnn_loss / n,
        "cnn_loss":      sum_cnn_loss / n,
    }

# ── Full test with per-class breakdown ─────────────────────────────────────────

def run_test_detailed(loader):
    gnn_model.eval(); cnn_model.eval()
    gnn_preds, gnn_targets = [], []
    cnn_preds, cnn_targets = [], []
    correct_vids = total_vids = 0
    frame_correct = frame_total = 0

    with torch.no_grad():
        for f, m, l in loader:
            for vid_idx in range(f.size(0)):
                video_frames = f[vid_idx]
                video_label  = l[vid_idx].to(device)
                video_masks  = m[vid_idx]

                (graph, _, all_logits, _,
                 fc, ft) = process_video_to_graph(
                    video_frames, video_masks, video_label,
                    cnn_model, base_vectors, device)

                total_vids    += 1
                frame_correct += fc
                frame_total   += ft

                if graph is not None:
                    gnn_pred = gnn_model(graph)
                    _, predicted = torch.max(gnn_pred, 1)
                    gnn_preds.extend(predicted.cpu().numpy())
                    gnn_targets.extend(video_label.unsqueeze(0).cpu().numpy())
                    correct_vids += (predicted == video_label.unsqueeze(0)).float().sum().item()

                if all_logits:
                    all_probs    = F.softmax(torch.cat(all_logits, dim=0), dim=1)
                    cnn_vid_pred = all_probs.mean(dim=0).argmax().item()
                    cnn_preds.append(cnn_vid_pred)
                    cnn_targets.append(video_label.item())

    gnn_cm        = confusion_matrix(gnn_targets, gnn_preds)
    cnn_cm        = confusion_matrix(cnn_targets, cnn_preds)
    gnn_per_class = gnn_cm.diagonal() / gnn_cm.sum(axis=1).clip(min=1)
    cnn_per_class = cnn_cm.diagonal() / cnn_cm.sum(axis=1).clip(min=1)
    gnn_acc       = correct_vids / max(total_vids, 1)
    cnn_frame_acc = frame_correct / max(frame_total, 1)
    return gnn_per_class, cnn_per_class, gnn_preds, gnn_targets, cnn_preds, cnn_targets, gnn_acc, cnn_frame_acc

# ── Diagnostics / plots ────────────────────────────────────────────────────────

def diagnose_zero_classes(gnn_per_class, cnn_per_class, gnn_preds, gnn_targets):
    zero_classes = [i for i, acc in enumerate(gnn_per_class) if acc == 0.0]
    print(f"\n{'─'*60}")
    print(f"GNN zero-accuracy classes ({len(zero_classes)}): {zero_classes}")
    for cls in zero_classes:
        wrong     = [gnn_preds[i] for i in range(len(gnn_targets)) if gnn_targets[i] == cls]
        count     = sum(1 for t in gnn_targets if t == cls)
        top_named = [(get_class_name(p), c) for p, c in Counter(wrong).most_common(3)]
        print(f"  Class {cls:2d} ({get_class_name(cls):12s}) | {count} test vids | "
              f"predicted as: {top_named}")
    both_zero = [i for i in range(num_classes)
                 if gnn_per_class[i] == 0.0 and cnn_per_class[i] == 0.0]
    print(f"\nClasses 0% in BOTH: {both_zero}")


def plot_gnn_vs_cnn_delta(gnn_per_class, cnn_per_class, test_dataset):
    counts     = Counter(int(l) for _, _, l in test_dataset)
    sizes      = np.array([counts[i] for i in range(num_classes)])
    heights    = 0.3 + (sizes / max(sizes)) * 0.7
    delta      = (gnn_per_class - cnn_per_class) * 100
    sorted_idx = np.argsort(delta)
    colors     = ['green' if d > 0 else 'steelblue' for d in delta[sorted_idx]]
    plt.figure(figsize=(9, 11))
    y_pos = np.arange(len(delta))
    plt.barh(y_pos, delta[sorted_idx], height=heights[sorted_idx], color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    max_abs = max(np.max(np.abs(delta)), 1)
    plt.xlim(-max_abs * 1.15, max_abs * 1.15)
    plt.yticks(y_pos, [get_class_name(i) for i in sorted_idx], fontsize=8)
    plt.xlabel("Δ Accuracy (%) (GNN − CNN)")
    plt.title(f"GNN vs CNN  [{SESSION_NAME} | aug={USE_AUGMENTATION} folds={K_FOLDS}]")
    for i, v in enumerate(delta[sorted_idx]):
        plt.text(v + (0.5 if v >= 0 else -3), i, f"{v:.1f}", va='center', fontsize=7)
    plt.tight_layout(); plt.savefig(spath("gnn_vs_cnn_delta.png"), dpi=120); plt.show()


def _tsne_embed_and_metrics(X, y, title):
    """Run t-SNE, compute metrics, return embedding + metrics string."""
    metrics_str = ""
    if len(np.unique(y)) >= 2:
        sil   = silhouette_score(X, y, metric='euclidean', sample_size=min(2000, len(X)))
        db    = davies_bouldin_score(X, y)
        ch    = calinski_harabasz_score(X, y)
        dists = pairwise_distances(X)
        same, diff = [], []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                (same if y[i]==y[j] else diff).append(dists[i,j])
        ratio = np.mean(diff) / max(np.mean(same), 1e-6)
        metrics_str = f"Sil:{sil:.3f} DB:{db:.3f} CH:{ch:.1f} Ratio:{ratio:.2f}x"
        print(f"\n  t-SNE '{title}': Sil={sil:.4f} DB={db:.4f} CH={ch:.1f} Ratio={ratio:.2f}x")
    X_emb = TSNE(n_components=2, perplexity=min(30, len(X)-1),
                 learning_rate=200, n_iter=1000, random_state=42).fit_transform(X)
    return X_emb, metrics_str


def plot_tsne(feats, lbls, title, fname):
    if len(feats) < 5:
        print(f"  t-SNE skipped ({len(feats)} samples)"); return
    X = np.stack(feats); y = np.array(lbls)
    X_emb, metrics_str = _tsne_embed_and_metrics(X, y, title)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(X_emb[:,0], X_emb[:,1], c=y, cmap='tab20', alpha=0.7, s=20)
    plt.colorbar(sc, ax=ax)
    ax.set_title(f"{title}\n{metrics_str}", fontsize=9)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.show()
    print(f"  Saved {fname}")


def plot_cnn_tsne_before_after(feats_before, lbls_before, feats_after, lbls_after, fname):
    """Side-by-side CNN backbone t-SNE: epoch 0 vs final epoch."""
    if len(feats_before) < 5 or len(feats_after) < 5:
        print("  before/after t-SNE skipped — not enough samples"); return

    X_b = np.stack(feats_before); y_b = np.array(lbls_before)
    X_a = np.stack(feats_after);  y_a = np.array(lbls_after)

    X_emb_b, ms_b = _tsne_embed_and_metrics(X_b, y_b, "CNN before GNN")
    X_emb_a, ms_a = _tsne_embed_and_metrics(X_a, y_a, "CNN after GNN")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, X_emb, y, ms, tag in [
        (axes[0], X_emb_b, y_b, ms_b, "Before GNN training (epoch 0)"),
        (axes[1], X_emb_a, y_a, ms_a, "After GNN training (final epoch)"),
    ]:
        sc = ax.scatter(X_emb[:,0], X_emb[:,1], c=y, cmap='tab20', alpha=0.7, s=20)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"CNN backbone features — {tag}\n{ms}", fontsize=9)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    plt.suptitle(f"CNN backbone t-SNE: effect of joint GNN training  [{SESSION_NAME}]",
                 fontsize=10)
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.show()
    print(f"  Saved {fname}")


def plot_training_curves(history, stopped_at, fold_idx):
    epochs = range(stopped_at)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Row 0: accuracies
    for ax, tk, vk, title in [
        (axes[0,0], "train_gnn_acc",       "val_gnn_acc",       "GNN Video Accuracy"),
        (axes[0,1], "train_cnn_vid_acc",   "val_cnn_vid_acc",   "CNN Video Accuracy"),
        (axes[0,2], "train_cnn_frame_acc", "val_cnn_frame_acc", "CNN Frame Accuracy"),
    ]:
        ax.plot(epochs, [x*100 for x in history[tk][:stopped_at]], label="Train")
        ax.plot(epochs, [x*100 for x in history[vk][:stopped_at]], label="Val")
        ax.set_title(title); ax.set_ylabel("Accuracy (%)"); ax.set_xlabel("Epoch")
        ax.legend()

    # Row 1: losses
    for ax, tk, vk, title in [
        (axes[1,0], "train_gnn_loss", "val_gnn_loss", "GNN Loss"),
        (axes[1,1], "train_cnn_loss", "val_cnn_loss", "CNN Loss"),
    ]:
        ax.plot(epochs, history[tk][:stopped_at], label="Train")
        ax.plot(epochs, history[vk][:stopped_at], label="Val")
        ax.set_title(title); ax.set_ylabel("Loss"); ax.set_xlabel("Epoch")
        ax.legend()

    axes[1,2].axis('off')   # spare cell

    plt.suptitle(f"[{SESSION_NAME}] aug={USE_AUGMENTATION} folds={K_FOLDS}  Fold {fold_idx+1}")
    plt.tight_layout()
    plt.savefig(spath(f"training_curves_fold{fold_idx+1}.png"), dpi=120); plt.show()


def pairwise_class_distances(feats, lbls, lbl=""):
    if len(feats) < 2: return
    X = np.stack(feats); y = np.array(lbls)
    dists = pairwise_distances(X)
    same, diff = [], []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            (same if y[i]==y[j] else diff).append(dists[i,j])
    print(f"  {lbl} — same: {np.mean(same):.4f}  diff: {np.mean(diff):.4f}  "
          f"ratio: {np.mean(diff)/max(np.mean(same),1e-6):.2f}x")

# ── Fixed test dataset ─────────────────────────────────────────────────────────
test_dataset = New_GNN_Dataset(test_files, split='test',
                               label_mapping=unique_label_mappings)
test_loader  = make_loader(test_dataset, shuffle=False)

global_best_val_loss  = float('inf')
global_best_gnn_state = None
global_best_cnn_state = None
fold_summaries        = []

# GNN embeddings — collected on last fold's final epoch only
tsne_gnn_feats, tsne_gnn_lbls = [], []
gnn_class_counts = defaultdict(int)

# CNN backbone features — collected TWICE per last fold: epoch 0 and final epoch
tsne_cnn_before_feats, tsne_cnn_before_lbls = [], []
tsne_cnn_after_feats,  tsne_cnn_after_lbls  = [], []
cnn_before_class_counts = defaultdict(int)
cnn_after_class_counts  = defaultdict(int)

training_start = time.time()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         FOLD TRAINING LOOP                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits):
    fold_train_files = [train_files[i] for i in tr_idx]
    fold_val_files   = [train_files[i] for i in val_idx]

    print(f"\n{'█'*65}")
    print(f"  FOLD {fold_idx+1}/{len(fold_splits)}  "
          f"mini-train={len(fold_train_files)}  val={len(fold_val_files)}  "
          f"aug={USE_AUGMENTATION}")
    print(f"{'█'*65}")

    train_split_tag = 'train' if USE_AUGMENTATION else 'test'

    fold_train_dataset = New_GNN_Dataset(fold_train_files, split=train_split_tag,
                                         label_mapping=unique_label_mappings)
    fold_val_dataset   = New_GNN_Dataset(fold_val_files,   split='test',
                                         label_mapping=unique_label_mappings)
    fold_train_loader  = make_loader(fold_train_dataset, shuffle=True)
    fold_val_loader    = make_loader(fold_val_dataset,   shuffle=False)

    cnn_model = FrameClassifierUnifiedCNN(num_classes=num_classes).to(device)
    trainable_cnn = [p for p in cnn_model.parameters() if p.requires_grad]
    print(f"  CNN trainable params: {sum(p.numel() for p in trainable_cnn):,} "
          f"(layer4 + head)")

    gnn_model = St_GCN_Classifier(num_classes=num_classes).to(device)

    # No class weights — plain label-smoothed CE
    cnn_lossfn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    gnn_lossfn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    gnn_optimiser = torch.optim.Adam(gnn_model.parameters(),
                                      lr=GNN_LR, weight_decay=GNN_WEIGHT_DECAY)
    cnn_optimiser = torch.optim.Adam(trainable_cnn,
                                      lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)

    gnn_scheduler = torch.optim.lr_scheduler.StepLR(
        gnn_optimiser, step_size=GNN_LR_STEP, gamma=GNN_LR_GAMMA)

    total_steps = len(fold_train_dataset) * NUM_EPOCHS
    cnn_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        cnn_optimiser, max_lr=CNN_LR, total_steps=total_steps,
        pct_start=0.2, anneal_strategy='cos',
        div_factor=10.0, final_div_factor=100.0,
    )

    history = {k: [] for k in [
        "train_gnn_acc",       "val_gnn_acc",
        "train_cnn_vid_acc",   "val_cnn_vid_acc",
        "train_cnn_frame_acc", "val_cnn_frame_acc",
        "train_gnn_loss",      "val_gnn_loss",
        "train_cnn_loss",      "val_cnn_loss",
    ]}

    early_stopper    = EarlyStopping(EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA)
    best_fold_loss   = float('inf')
    best_fold_gnn    = None
    best_fold_cnn    = None
    stopped_at_epoch = NUM_EPOCHS
    is_last_fold     = (fold_idx == len(fold_splits) - 1)

    for epoch in range(NUM_EPOCHS):
        print(f"\n  {'─'*40} FOLD {fold_idx+1} EPOCH {epoch}/{NUM_EPOCHS-1}")
        t0 = time.time()

        # Collect CNN t-SNE BEFORE training at epoch 0 (last fold only)
        collect_cnn_before = is_last_fold and epoch == 0
        # Collect CNN + GNN t-SNE at final epoch (last fold only)
        collect_final = is_last_fold and (
            epoch == NUM_EPOCHS - 1 or early_stopper.should_stop)

        train_metrics = run_epoch(
            fold_train_loader, train=True,
            # CNN before: epoch 0, no GNN gradients have touched backbone yet
            tsne_cnn_feats=tsne_cnn_before_feats if collect_cnn_before else (
                           tsne_cnn_after_feats  if collect_final else None),
            tsne_cnn_lbls =tsne_cnn_before_lbls  if collect_cnn_before else (
                           tsne_cnn_after_lbls   if collect_final else None),
            cnn_class_counts=cnn_before_class_counts if collect_cnn_before else (
                             cnn_after_class_counts  if collect_final else defaultdict(int)),
            tsne_gnn_feats=tsne_gnn_feats if collect_final else None,
            tsne_gnn_lbls =tsne_gnn_lbls  if collect_final else None,
            gnn_class_counts=gnn_class_counts if collect_final else defaultdict(int),
        )
        val_metrics = run_epoch(fold_val_loader, train=False)

        gnn_scheduler.step()

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        # map run_epoch keys → history keys
        history["val_gnn_acc"].append(val_metrics["gnn_acc"])
        history["val_cnn_vid_acc"].append(val_metrics["cnn_vid_acc"])
        history["val_cnn_frame_acc"].append(val_metrics["cnn_frame_acc"])
        history["val_gnn_loss"].append(val_metrics["gnn_loss"])
        history["val_cnn_loss"].append(val_metrics["cnn_loss"])

        checkpoint_loss = val_metrics['gnn_loss'] + 0.8 * val_metrics['cnn_loss']

        print(f"  Train — GNN: {100*train_metrics['gnn_acc']:.1f}%  "
              f"CNN vid: {100*train_metrics['cnn_vid_acc']:.1f}%  "
              f"CNN frame: {100*train_metrics['cnn_frame_acc']:.1f}%  "
              f"GNN loss: {train_metrics['gnn_loss']:.4f}  "
              f"CNN loss: {train_metrics['cnn_loss']:.4f}")
        print(f"  Val   — GNN: {100*val_metrics['gnn_acc']:.1f}%  "
              f"CNN vid: {100*val_metrics['cnn_vid_acc']:.1f}%  "
              f"CNN frame: {100*val_metrics['cnn_frame_acc']:.1f}%  "
              f"ckpt loss: {checkpoint_loss:.4f}  ({time.time()-t0:.0f}s)")

        if checkpoint_loss < best_fold_loss:
            best_fold_loss = checkpoint_loss
            best_fold_gnn  = {k: v.clone() for k, v in gnn_model.state_dict().items()}
            best_fold_cnn  = {k: v.clone() for k, v in cnn_model.state_dict().items()}
            print(f"  → Fold checkpoint saved (loss {best_fold_loss:.4f})")

        early_stopper.step(checkpoint_loss)
        if early_stopper.should_stop:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            stopped_at_epoch = epoch + 1
            # If we haven't collected final t-SNE yet, do it now
            if is_last_fold and not collect_final:
                run_epoch(fold_train_loader, train=False,
                          tsne_cnn_feats=tsne_cnn_after_feats,
                          tsne_cnn_lbls=tsne_cnn_after_lbls,
                          cnn_class_counts=cnn_after_class_counts,
                          tsne_gnn_feats=tsne_gnn_feats,
                          tsne_gnn_lbls=tsne_gnn_lbls,
                          gnn_class_counts=gnn_class_counts)
            break

    if best_fold_loss < global_best_val_loss:
        global_best_val_loss  = best_fold_loss
        global_best_gnn_state = best_fold_gnn
        global_best_cnn_state = best_fold_cnn
        print(f"\n  ★ New global best Fold {fold_idx+1} "
              f"(val loss {global_best_val_loss:.4f})")

    plot_training_curves(history, stopped_at_epoch, fold_idx)
    fold_summaries.append({
        "fold": fold_idx+1, "best_val_loss": round(best_fold_loss, 4),
        "stopped_epoch": stopped_at_epoch,
        "val_gnn_acc":       round(val_metrics['gnn_acc'], 4),
        "val_cnn_vid_acc":   round(val_metrics['cnn_vid_acc'], 4),
        "val_cnn_frame_acc": round(val_metrics['cnn_frame_acc'], 4),
    })

print(f"\nAll folds complete — {(time.time()-training_start)/60:.1f}min total")
for s in fold_summaries:
    print(f"  Fold {s['fold']}: val_loss={s['best_val_loss']:.4f}  "
          f"GNN={100*s['val_gnn_acc']:.1f}%  "
          f"CNN-vid={100*s['val_cnn_vid_acc']:.1f}%  "
          f"CNN-frame={100*s['val_cnn_frame_acc']:.1f}%  "
          f"stopped@ep{s['stopped_epoch']}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   RESTORE BEST & EVALUATE ON TEST                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

cnn_model = FrameClassifierUnifiedCNN(num_classes=num_classes).to(device)
gnn_model = St_GCN_Classifier(num_classes=num_classes).to(device)

full_train_dataset = New_GNN_Dataset(train_files, split='test',
                                     label_mapping=unique_label_mappings)
# No class weights
cnn_lossfn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
gnn_lossfn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

gnn_model.load_state_dict(global_best_gnn_state)
cnn_model.load_state_dict(global_best_cnn_state)
torch.save(global_best_gnn_state, spath("gnn_best.pt"))
torch.save(global_best_cnn_state, spath("cnn_best.pt"))
print(f"\nRestored global best (val loss {global_best_val_loss:.4f})")

print("\n" + "="*60 + "\nFINAL TEST EVALUATION\n" + "="*60)

(gnn_per_class, cnn_per_class,
 gnn_preds, gnn_targets,
 cnn_preds, cnn_targets,
 final_gnn_acc, final_cnn_frame_acc) = run_test_detailed(test_loader)

cnn_video_acc = sum(p==t for p,t in zip(cnn_preds,cnn_targets)) / max(len(cnn_preds),1)
print(f"\nGNN Video Accuracy  : {100*final_gnn_acc:.1f}%")
print(f"CNN Video Accuracy  : {100*cnn_video_acc:.1f}%")
print(f"CNN Frame Accuracy  : {100*final_cnn_frame_acc:.1f}%")

print("\nPer-class accuracy (GNN | CNN):")
for i in range(num_classes):
    print(f"  Class {i:2d} ({get_class_name(i):12s}):  "
          f"GNN {100*gnn_per_class[i]:.1f}%  |  CNN {100*cnn_per_class[i]:.1f}%")

print("\nPrediction distribution (GNN):", Counter(gnn_preds))
print("Target distribution:         ", Counter(gnn_targets))

print("\n" + "="*60 + "\nDIAGNOSTICS\n" + "="*60)
diagnose_zero_classes(gnn_per_class, cnn_per_class, gnn_preds, gnn_targets)

print("\nChecking for overfitting...")
full_train_loader = make_loader(full_train_dataset, shuffle=False)
train_detailed    = run_epoch(full_train_loader, train=False)
gap = train_detailed['gnn_acc'] - final_gnn_acc
print(f"  GNN train: {100*train_detailed['gnn_acc']:.1f}%  "
      f"test: {100*final_gnn_acc:.1f}%  gap: {100*gap:.1f}%")
print("  ⚠ Large generalisation gap" if gap > 0.3 else "  ✓ Reasonable generalisation gap")

print("\nFeature space pairwise distances:")
pairwise_class_distances(tsne_gnn_feats, tsne_gnn_lbls, "GNN embeddings")
pairwise_class_distances(tsne_cnn_after_feats, tsne_cnn_after_lbls, "CNN backbone (after)")
pairwise_class_distances(tsne_cnn_before_feats, tsne_cnn_before_lbls, "CNN backbone (before)")

plot_gnn_vs_cnn_delta(gnn_per_class, cnn_per_class, test_dataset)

# Before/after CNN t-SNE — the key diagnostic plot
plot_cnn_tsne_before_after(
    tsne_cnn_before_feats, tsne_cnn_before_lbls,
    tsne_cnn_after_feats,  tsne_cnn_after_lbls,
    spath("cnn_tsne_before_after.png")
)

# Individual t-SNE plots
mode_tag = "SYNTHETIC" if SYNTHETIC_MODE else "Real"
plot_tsne(tsne_cnn_after_feats, tsne_cnn_after_lbls,
          f"CNN backbone (after GNN) [{mode_tag} | aug={USE_AUGMENTATION}]",
          spath("cnn_tsne_after.png"))
plot_tsne(tsne_cnn_before_feats, tsne_cnn_before_lbls,
          f"CNN backbone (before GNN, epoch 0) [{mode_tag}]",
          spath("cnn_tsne_before.png"))
plot_tsne(tsne_gnn_feats, tsne_gnn_lbls,
          f"GNN embeddings [{mode_tag} | aug={USE_AUGMENTATION}]",
          spath("gnn_tsne_final.png"))

for values, val_values, ylabel, fname in [
    (history["train_gnn_loss"], history["val_gnn_loss"],
     "GNN Loss", spath("gnn_loss.png")),
    (history["train_cnn_loss"], history["val_cnn_loss"],
     "CNN Loss", spath("cnn_loss.png")),
    (history["train_cnn_frame_acc"], history["val_cnn_frame_acc"],
     "CNN Frame Accuracy", spath("cnn_frame_acc.png")),
]:
    _, ax = plt.subplots(1, 1)
    scale = 100 if "Accuracy" in ylabel else 1
    ax.plot([x*scale for x in values[:stopped_at_epoch]], label="Train")
    ax.plot([x*scale for x in val_values[:stopped_at_epoch]], label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  [aug={USE_AUGMENTATION} folds={K_FOLDS}]"); ax.legend()
    plt.savefig(fname); plt.show()

results_summary = {
    "session_name": SESSION_NAME, "session_notes": SESSION_NOTES,
    "use_augmentation": USE_AUGMENTATION, "k_folds": K_FOLDS,
    "final_gnn_acc":        float(final_gnn_acc),
    "final_cnn_video_acc":  float(cnn_video_acc),
    "final_cnn_frame_acc":  float(final_cnn_frame_acc),
    "global_best_val_loss": float(global_best_val_loss),
    "train_time_min": round((time.time() - training_start) / 60, 1),
    "fold_summaries": fold_summaries,
    "gnn_per_class_acc": [round(float(x), 4) for x in gnn_per_class],
    "cnn_per_class_acc": [round(float(x), 4) for x in cnn_per_class],
}
with open(spath("results.json"), "w") as fp:
    json.dump(results_summary, fp, indent=2)

print(f"\n{'█'*65}")
print(f"  SESSION {SESSION_NAME} COMPLETE")
print(f"  aug={USE_AUGMENTATION}  folds={K_FOLDS}")
print(f"  GNN: {100*final_gnn_acc:.1f}%   "
      f"CNN-vid: {100*cnn_video_acc:.1f}%   "
      f"CNN-frame: {100*final_cnn_frame_acc:.1f}%   "
      f"Time: {results_summary['train_time_min']:.1f}min")
print(f"  Outputs → {OUT_DIR}/")
print("█" * 65)