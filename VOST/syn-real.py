# """
# CNN + GNN training script — fixed version.

# Key fixes applied:
#   1. Dropout re-added to GNN (was removed, causing overfitting)
#   2. Weight decay (L2) on both optimisers
#   3. Label smoothing on loss functions
#   4. Early stopping on val GNN loss
#   5. GNN hidden dim reduced 256 → 128 (less capacity for tiny dataset)
#   6. CNN backbone frozen — only classifier head fine-tuned
#   7. Val checkpoint pass de-duplicated (no longer runs test set twice)
#   8. Separate class_counts for CNN and GNN t-SNE collection
#   9. CNN t-SNE collects backbone feature maps, not logits
#   10. Fallback node counter added for diagnostics
# """

# import os, json, time, math, random
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from collections import defaultdict, Counter
# from sklearn.manifold import TSNE
# from sklearn.metrics import confusion_matrix, pairwise_distances, silhouette_score
# from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
# from torch_geometric.nn import GCNConv, global_mean_pool
# from torch_geometric.data import Data
# from torch.utils.data import DataLoader as TorchDataLoader
# from skimage.measure import label, regionprops

# from cnn_baseline import FrameClassifierUnifiedCNN
# from gnn_dataset import New_GNN_Dataset
# from utils import get_object_label, extract_label_from_filename, get_file_names

# # ╔══════════════════════════════════════════════════════════════════════════╗
# # ║                       SESSION CONFIGURATION                             ║
# # ╚══════════════════════════════════════════════════════════════════════════╝

# SESSION_NAME  = "s4-spatial-only-lambda-0.2-data-aug"
# SESSION_NOTES = (
#     "Dropout re-added, weight decay, label smoothing, early stopping, "
#     "frozen CNN backbone, reduced GNN dim 256→128"
# )

# # ── Model / training hyperparameters ───────────────────────────────────────
# GNN_LR              = 1e-3
# CNN_LR              = 1e-4          # only the CNN head is trained
# NUM_EPOCHS          = 20            # early stopping will cut this short
# CHUNK_SIZE          = 30
# GNN_LR_STEP         = 10
# GNN_LR_GAMMA        = 0.5
# GNN_WEIGHT_DECAY    = 1e-3          # L2 regularisation on GNN
# CNN_WEIGHT_DECAY    = 1e-4          # L2 regularisation on CNN head
# LABEL_SMOOTHING     = 0.1           # smooths one-hot targets → less overconfident
# GNN_DROPOUT         = 0.3           # dropout on GNN node embeddings
# GNN_HIDDEN_DIM      = 128           # reduced from 256
# INTRA_FRAME_EDGES   = True
# MIN_REGION_AREA     = 50
# LAMBDA              = 0.2           # weight for GNN loss when combined with CNN loss

# # ── Early stopping ──────────────────────────────────────────────────────────
# EARLY_STOP_PATIENCE = 5            # stop if val loss doesn't improve for N epochs
# EARLY_STOP_MIN_DELTA = 0.03         # minimum improvement to count as progress

# # ── Feature / graph config ─────────────────────────────────────────────────
# SYNTHETIC_MODE        = False
# SYNTHETIC_NOISE_SCALE = 0.1
# SYNTHETIC_FEAT_DIM    = 2048
# MAX_PER_CLASS_TSNE    = 70
# USE_SPLITS_JSON       = True

# # ── Output directory ────────────────────────────────────────────────────────
# OUT_DIR = f"outputs/{SESSION_NAME}"
# os.makedirs(OUT_DIR, exist_ok=True)

# def spath(filename):
#     return os.path.join(OUT_DIR, filename)

# # ── Print session banner ────────────────────────────────────────────────────
# print("\n" + "█" * 65)
# print(f"  SESSION : {SESSION_NAME}")
# print(f"  NOTES   : {SESSION_NOTES}")
# print(f"  GNN LR={GNN_LR}  WD={GNN_WEIGHT_DECAY}  Dropout={GNN_DROPOUT}  "
#       f"Hidden={GNN_HIDDEN_DIM}")
# print(f"  CNN LR={CNN_LR}  WD={CNN_WEIGHT_DECAY}  LabelSmooth={LABEL_SMOOTHING}")
# print(f"  Early stop patience={EARLY_STOP_PATIENCE}  Epochs={NUM_EPOCHS}")
# print(f"  Outputs → {OUT_DIR}/")
# print("█" * 65 + "\n")

# session_config = dict(
#     session_name=SESSION_NAME, session_notes=SESSION_NOTES,
#     gnn_lr=GNN_LR, cnn_lr=CNN_LR, num_epochs=NUM_EPOCHS,
#     gnn_weight_decay=GNN_WEIGHT_DECAY, cnn_weight_decay=CNN_WEIGHT_DECAY,
#     label_smoothing=LABEL_SMOOTHING, gnn_dropout=GNN_DROPOUT,
#     gnn_hidden_dim=GNN_HIDDEN_DIM, early_stop_patience=EARLY_STOP_PATIENCE,
#     intra_frame_edges=INTRA_FRAME_EDGES, synthetic_mode=SYNTHETIC_MODE,
# )
# with open(spath("config.json"), "w") as fp:
#     json.dump(session_config, fp, indent=2)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA']   = "1"

# # ── Load splits ─────────────────────────────────────────────────────────────
# MIN_SAMPLES = 10
# OUTPUT_FILE = f"splits_{MIN_SAMPLES}.json"

# if USE_SPLITS_JSON:
#     with open(OUTPUT_FILE) as fp:
#         splits = json.load(fp)
#     train_files           = splits["train"]
#     test_files            = splits["test"]
#     unique_label_mappings = splits["label_mapping"]
#     num_classes           = splits["num_classes"]
#     val_files             = splits.get("val", [])
# else:
#     all_valid_file_names = ['455_fold_box', '4320_tear_dough', '226_squeeze_dough',
#                             '3705_flatten_box', '4176_cut_cloth', '4174_cut_cloth',
#                             '5182_open_box', '8822_separate_dough']
#     MIN_SAMPLES, MAX_SAMPLES, TEST_SPLIT = 0, 50, 0.2
#     class_to_files = defaultdict(list)
#     for f in all_valid_file_names:
#         class_to_files[get_object_label(f)].append(f)
#     filtered = {cls: files[:MAX_SAMPLES] for cls, files in class_to_files.items()
#                 if len(files) >= MIN_SAMPLES}
#     train_files, test_files, val_files = [], [], []
#     for cls, files in filtered.items():
#         random.shuffle(files)
#         n = len(files)
#         n_test = max(1, int(n * TEST_SPLIT))
#         train_files.extend(files[:-n_test])
#         test_files.extend(files[-n_test:])
#     all_files  = train_files + test_files
#     all_labels = sorted({extract_label_from_filename(f) for f in all_files})
#     unique_label_mappings = {lbl: idx for idx, lbl in enumerate(all_labels)}
#     num_classes = len(unique_label_mappings)

# print(f"Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")
# print(f"Classes: {num_classes}  |  {unique_label_mappings}\n")

# # ── Helpers ─────────────────────────────────────────────────────────────────

# def moving_average(data, window=5):
#     return np.convolve(data, np.ones(window) / window, mode='valid')

# def make_sinusoidal_base_vectors(num_classes, d_model):
#     pe       = torch.zeros(num_classes, d_model)
#     position = torch.arange(num_classes, dtype=torch.float).unsqueeze(1)
#     div_term = torch.exp(
#         torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
#     )
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
#     return pe

# def synthetic_node_feature(class_idx, base_vectors, noise_scale, device):
#     base = base_vectors[class_idx].clone()
#     feat = (base + torch.randn_like(base) * noise_scale).to(device)
#     return feat.unsqueeze(-1).unsqueeze(-1)

# def mask_to_feat_soft_area(mask_hw, feat):
#     _, HF, WF = feat.shape
#     m = mask_hw.float()[None, None, ...]
#     return F.interpolate(m, size=(HF, WF), mode="area")

# def masked_avrg_pool(mask_feat, feat_map):
#     return (mask_feat * feat_map).sum(dim=(2, 3)) / mask_feat.sum().clamp_min(1e-6)

# def pool_object_vec(feat_map, mask):
#     """Returns (C, 1, 1)."""
#     mask      = torch.as_tensor(mask, dtype=torch.float32, device=feat_map.device)
#     mask_feat = mask_to_feat_soft_area(mask, feat_map)
#     obj_vec   = masked_avrg_pool(mask_feat, feat_map.unsqueeze(0))
#     return obj_vec.squeeze(0).unsqueeze(-1).unsqueeze(-1)

# def get_class_weights(dataset, num_classes, device):
#     counts = torch.zeros(num_classes)
#     for _, _, l in dataset:
#         counts[l] += 1
#     w = 1.0 / (counts + 1e-6)
#     return (w / w.sum() * num_classes).to(device)

# def get_class_name(idx):
#     return [k for k, v in unique_label_mappings.items() if v == idx][0]

# # ── GNN model — dropout restored, hidden dim reduced ────────────────────────

# class St_GCN_Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         H = GNN_HIDDEN_DIM
#         self.node_proj = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(SYNTHETIC_FEAT_DIM, H),
#             nn.ReLU(),
#             nn.Dropout(p=GNN_DROPOUT),        # FIX 1: dropout re-added
#         )
#         self.conv1      = GCNConv(H, H)
#         self.drop       = nn.Dropout(p=GNN_DROPOUT)
#         self.classifier = nn.Linear(H, num_classes)

#     def forward(self, data, return_hidden=False):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x      = self.node_proj(x)
#         x      = x + self.drop(F.relu(self.conv1(x, edge_index)))  # residual + dropout
#         pooled = global_mean_pool(x, batch)
#         if return_hidden:
#             return self.classifier(pooled), pooled
#         return self.classifier(pooled)


# # ── Graph building ───────────────────────────────────────────────────────────

# def build_graph(subimages_per_frame, device):
#     frame_ids     = sorted(subimages_per_frame.keys())
#     frame_offsets = {}
#     offset = 0
#     for fid in frame_ids:
#         frame_offsets[fid] = offset
#         offset += len(subimages_per_frame[fid])

#     all_nodes, edges_from, edges_to = [], [], []

#     for idx, fid in enumerate(frame_ids):
#         nodes_here = subimages_per_frame[fid]
#         all_nodes.extend(nodes_here)

#         if INTRA_FRAME_EDGES:
#             base = frame_offsets[fid]
#             for ci in range(len(nodes_here)):
#                 for cj in range(len(nodes_here)):
#                     if ci != cj:
#                         edges_from.append(base + ci)
#                         edges_to.append(base + cj)

#         if idx == len(frame_ids) - 1:
#             continue
#         # next_fid = frame_ids[idx + 1]
#         # for ci in range(len(nodes_here)):
#         #     src = frame_offsets[fid] + ci
#         #     for ni in range(len(subimages_per_frame[next_fid])):
#         #         edges_from.append(src)
#         #         edges_to.append(frame_offsets[next_fid] + ni)

#     if not all_nodes:
#         return None

#     nodes_tensor = torch.stack(all_nodes)
#     edge_index   = torch.tensor([edges_from, edges_to], dtype=torch.long).to(device)
#     return Data(x=nodes_tensor, edge_index=edge_index).to(device)


# def process_video_to_graph(video_frames, video_masks, video_label,
#                             cnn_model, base_vectors, device):
#     """
#     Returns (graph, avg_cnn_loss, all_logits, backbone_feat_vec).
#     backbone_feat_vec: (2048,) global avg-pooled backbone feature for t-SNE.
#                        None in SYNTHETIC_MODE.
#     """
#     F_len               = video_frames.shape[0]
#     subimages_per_frame = {}
#     global_frame_count  = 0
#     allcnnlosses        = torch.tensor(0.0, device=device)
#     all_logits          = []
#     all_feat_vecs       = []   # FIX 9: backbone features, not logits
#     fallback_count      = 0    # FIX 10: diagnostic counter

#     for i in range(0, F_len, CHUNK_SIZE):
#         chunk = video_frames[i:i + CHUNK_SIZE].to(device)
#         chunk = F.interpolate(chunk, size=(224, 224),
#                               mode='bilinear', align_corners=False)

#         if not SYNTHETIC_MODE:
#             logits, feature_maps = cnn_model(chunk)   # feature_maps: (B, 2048, H, W)
#             all_logits.append(logits)
#             # Global avg pool backbone features → (B, 2048) then mean over chunk → (2048,)
#             feat_vec = feature_maps.mean(dim=(2, 3))  # (B, 2048)
#             all_feat_vecs.append(feat_vec.detach())
#             video_labels_chunk = video_label.repeat(chunk.size(0))
#             allcnnlosses = allcnnlosses + cnn_lossfn(logits, video_labels_chunk)
#         else:
#             logits, feature_maps = cnn_model(chunk)
#             allcnnlosses = torch.tensor(0.0, requires_grad=True, device=device)

#         for j in range(chunk.size(0)):
#             frame_idx = i + j
#             subimages_per_frame[global_frame_count] = []

#             if SYNTHETIC_MODE:
#                 node_feat = synthetic_node_feature(
#                     int(video_label.item()), base_vectors,
#                     SYNTHETIC_NOISE_SCALE, device)
#                 subimages_per_frame[global_frame_count].append(node_feat)
#                 global_frame_count += 1
#             else:
#                 mask    = video_masks[frame_idx].float()
#                 mask_np = mask.cpu().numpy().astype("uint8")
#                 regions = regionprops(label(mask_np, connectivity=2))
#                 H, W    = mask_np.shape

#                 for r in regions:
#                     if r.area <= MIN_REGION_AREA:
#                         continue
#                     rm = torch.zeros((H, W), dtype=torch.float32)
#                     rm[r.coords[:, 0], r.coords[:, 1]] = 1.0
#                     node_vec = pool_object_vec(feature_maps[j], rm.to(device))
#                     subimages_per_frame[global_frame_count].append(node_vec)

#                 # Skip frame entirely if no valid regions found
#                 if not subimages_per_frame[global_frame_count]:
#                     fallback_count += 1
#                     del subimages_per_frame[global_frame_count]
#                     # do NOT increment global_frame_count — frame is dropped
#                     continue

#                 global_frame_count += 1

#     if fallback_count > 0:
#         print(f"{fallback_count}/{F_len} frames skipped (no valid regions)")

#     graph        = build_graph(subimages_per_frame, device)
#     num_chunks   = max(1, F_len // CHUNK_SIZE)
#     avg_cnn_loss = allcnnlosses / num_chunks

#     # Aggregate backbone features across all chunks → single (2048,) vector
#     backbone_feat_vec = None
#     if all_feat_vecs:
#         backbone_feat_vec = torch.cat(all_feat_vecs, dim=0).mean(dim=0)  # (2048,)

#     return graph, avg_cnn_loss, all_logits, backbone_feat_vec


# # ── Early stopping helper ────────────────────────────────────────────────────

# class EarlyStopping:
#     def __init__(self, patience, min_delta):
#         self.patience   = patience
#         self.min_delta  = min_delta
#         self.best_loss  = float('inf')
#         self.counter    = 0
#         self.should_stop = False

#     def step(self, val_loss):
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter   = 0
#         else:
#             self.counter += 1
#             print(f"  Early stop counter: {self.counter}/{self.patience}")
#             if self.counter >= self.patience:
#                 self.should_stop = True


# # ── Epoch runner ─────────────────────────────────────────────────────────────

# def run_epoch(loader, train=True,
#               tsne_cnn_feats=None, tsne_cnn_lbls=None, cnn_class_counts=None,
#               tsne_gnn_feats=None, tsne_gnn_lbls=None, gnn_class_counts=None):
#     """
#     One full pass over loader.
#     FIX 8: separate cnn_class_counts and gnn_class_counts.
#     FIX 9: collects backbone feature maps for CNN t-SNE, not logits.
#     """
#     if train:
#         gnn_model.train(); cnn_model.train()
#     else:
#         gnn_model.eval();  cnn_model.eval()

#     total_vids = correct_gnn = correct_cnn_vids = 0
#     sum_gnn_loss = sum_cnn_loss = 0.0

#     ctx = torch.enable_grad() if train else torch.no_grad()
#     with ctx:
#         for f, m, l in loader:
#             # f = f.float() / 255.0
#             # f = f.permute(0, 1, 4, 2, 3)

#             for vid_idx in range(f.size(0)):
#                 video_frames = f[vid_idx]
#                 video_label  = l[vid_idx].to(device)
#                 video_masks  = m[vid_idx]

#                 if train:
#                     gnn_optimiser.zero_grad()
#                     cnn_optimiser.zero_grad()

#                 graph, avg_cnn_loss, all_logits, backbone_feat = process_video_to_graph(
#                     video_frames, video_masks, video_label,
#                     cnn_model, base_vectors, device)

#                 if graph is None:
#                     continue

#                 gnn_pred, hidden = gnn_model(graph, return_hidden=True)
#                 gnn_loss = gnn_lossfn(gnn_pred, video_label.unsqueeze(0))

#                 if train:
#                     total_loss = gnn_loss if SYNTHETIC_MODE else avg_cnn_loss + LAMBDA * gnn_loss ## add lambda
#                     total_loss.backward()
#                     torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), 1.0)
#                     torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
#                     gnn_optimiser.step()
#                     if not SYNTHETIC_MODE:
#                         cnn_optimiser.step()

#                 total_vids   += 1
#                 correct_gnn  += (gnn_pred.argmax(1) == video_label.unsqueeze(0)).float().sum().item()
#                 sum_gnn_loss += gnn_loss.item()
#                 sum_cnn_loss += avg_cnn_loss.item() if not SYNTHETIC_MODE else 0.0

#                 # CNN video-level accuracy: average softmax across all frames, take argmax
#                 if all_logits:
#                     all_probs   = F.softmax(torch.cat(all_logits, dim=0), dim=1)  # (F, C)
#                     avg_prob    = all_probs.mean(dim=0)                            # (C,)
#                     cnn_pred_vid = avg_prob.argmax().item()
#                     correct_cnn_vids += int(cnn_pred_vid == video_label.item())

#                 lbl_int = int(video_label.item())

#                 # FIX 8+9: CNN t-SNE from backbone features, own counter
#                 if (tsne_cnn_feats is not None and backbone_feat is not None
#                         and cnn_class_counts[lbl_int] < MAX_PER_CLASS_TSNE):
#                     tsne_cnn_feats.append(backbone_feat.cpu().numpy().astype(np.float32))
#                     tsne_cnn_lbls.append(lbl_int)
#                     cnn_class_counts[lbl_int] += 1

#                 # GNN t-SNE from pooled embedding, own counter
#                 if (tsne_gnn_feats is not None
#                         and gnn_class_counts[lbl_int] < MAX_PER_CLASS_TSNE):
#                     tsne_gnn_feats.append(
#                         hidden.detach().cpu().numpy().astype(np.float32).squeeze(0))
#                     tsne_gnn_lbls.append(lbl_int)
#                     gnn_class_counts[lbl_int] += 1

#     n = max(total_vids, 1)
#     return {
#         "gnn_acc":  correct_gnn / n,
#         "cnn_acc":  correct_cnn_vids / n,   # video-level softmax vote
#         "gnn_loss": sum_gnn_loss / n,
#         "cnn_loss": sum_cnn_loss / n,
#     }


# # ── Full test with per-class breakdown ───────────────────────────────────────

# def run_test_detailed(loader):
#     gnn_model.eval(); cnn_model.eval()
#     gnn_preds, gnn_targets = [], []
#     cnn_preds, cnn_targets = [], []
#     correct_vids = total_vids = 0

#     with torch.no_grad():
#         for f, m, l in loader:
#             f = f.float() / 255.0
#             f = f.permute(0, 1, 4, 2, 3)
#             for vid_idx in range(f.size(0)):
#                 video_frames = f[vid_idx]
#                 video_label  = l[vid_idx].to(device)
#                 video_masks  = m[vid_idx]

#                 graph, _, all_logits, _ = process_video_to_graph(
#                     video_frames, video_masks, video_label,
#                     cnn_model, base_vectors, device)

#                 total_vids += 1

#                 if graph is not None:
#                     gnn_pred = gnn_model(graph)
#                     _, predicted = torch.max(gnn_pred, 1)
#                     gnn_preds.extend(predicted.cpu().numpy())
#                     gnn_targets.extend(video_label.unsqueeze(0).cpu().numpy())
#                     correct_vids += (predicted == video_label.unsqueeze(0)).float().sum().item()

#                 # CNN video-level: softmax average across all frames → argmax
#                 if all_logits:
#                     all_probs    = F.softmax(torch.cat(all_logits, dim=0), dim=1)  # (F, C)
#                     cnn_vid_pred = all_probs.mean(dim=0).argmax().item()
#                     cnn_preds.append(cnn_vid_pred)
#                     cnn_targets.append(video_label.item())

#     gnn_cm = confusion_matrix(gnn_targets, gnn_preds)
#     cnn_cm = confusion_matrix(cnn_targets, cnn_preds)
#     gnn_per_class = gnn_cm.diagonal() / gnn_cm.sum(axis=1).clip(min=1)
#     cnn_per_class = cnn_cm.diagonal() / cnn_cm.sum(axis=1).clip(min=1)
#     gnn_acc = correct_vids / max(total_vids, 1)
#     return gnn_per_class, cnn_per_class, gnn_preds, gnn_targets, cnn_preds, cnn_targets, gnn_acc


# # ── Diagnostics ──────────────────────────────────────────────────────────────

# def diagnose_zero_classes(gnn_per_class, cnn_per_class, gnn_preds, gnn_targets):
#     zero_classes = [i for i, acc in enumerate(gnn_per_class) if acc == 0.0]
#     print(f"\n{'─'*60}")
#     print(f"GNN zero-accuracy classes ({len(zero_classes)}): {zero_classes}")
#     for cls in zero_classes:
#         wrong    = [gnn_preds[i] for i in range(len(gnn_targets)) if gnn_targets[i] == cls]
#         count    = sum(1 for t in gnn_targets if t == cls)
#         name     = get_class_name(cls)
#         top      = Counter(wrong).most_common(3)
#         top_named = [(get_class_name(p), c) for p, c in top]
#         print(f"  Class {cls:2d} ({name:12s}) | {count} test vids | "
#               f"predicted as: {top_named}")

#     both_zero = [i for i in range(num_classes)
#                  if gnn_per_class[i] == 0.0 and cnn_per_class[i] == 0.0]
#     print(f"\nClasses 0% in BOTH: {both_zero}")

#     cnn_good_gnn_bad = [i for i in range(num_classes)
#                         if gnn_per_class[i] == 0.0 and cnn_per_class[i] > 0.3]
#     print(f"CNN>30% but GNN 0%: {[(c, get_class_name(c)) for c in cnn_good_gnn_bad]}")


# def plot_gnn_vs_cnn_delta(gnn_per_class, cnn_per_class, test_dataset):
#     counts     = Counter(int(l) for _, _, l in test_dataset)
#     sizes      = np.array([counts[i] for i in range(num_classes)])
#     heights    = 0.3 + (sizes / max(sizes)) * 0.7
#     delta      = (gnn_per_class - cnn_per_class) * 100
#     sorted_idx = np.argsort(delta)
#     delta_s    = delta[sorted_idx]
#     heights_s  = heights[sorted_idx]
#     names_s    = [get_class_name(i) for i in sorted_idx]
#     colors     = ['green' if d > 0 else 'steelblue' for d in delta_s]

#     plt.figure(figsize=(9, 11))
#     y_pos = np.arange(len(delta_s))
#     plt.barh(y_pos, delta_s, height=heights_s, color=colors)
#     plt.axvline(0, color='black', linewidth=0.8)
#     max_abs = max(np.max(np.abs(delta_s)), 1)
#     plt.xlim(-max_abs * 1.15, max_abs * 1.15)
#     plt.yticks(y_pos, names_s, fontsize=8)
#     plt.xlabel("Δ Accuracy (%) (GNN − CNN)")
#     plt.title(f"GNN vs CNN per Class  [{SESSION_NAME}]\n(bar thickness = test set size)")
#     for i, v in enumerate(delta_s):
#         plt.text(v + (0.5 if v >= 0 else -3), i, f"{v:.1f}", va='center', fontsize=7)
#     plt.tight_layout()
#     plt.savefig(spath("gnn_vs_cnn_delta.png"), dpi=120)
#     plt.show()


# def plot_tsne(feats, lbls, title, fname):
#     if len(feats) < 5:
#         print(f"  t-SNE skipped — not enough samples ({len(feats)})")
#         return
#     X = np.stack(feats)
#     y = np.array(lbls)

#     metrics_str = ""
#     if len(np.unique(y)) >= 2:
#         sil   = silhouette_score(X, y, metric='euclidean', sample_size=min(2000, len(X)))
#         db    = davies_bouldin_score(X, y)
#         ch    = calinski_harabasz_score(X, y)
#         dists = pairwise_distances(X)
#         same, diff = [], []
#         for i in range(len(X)):
#             for j in range(i + 1, len(X)):
#                 (same if y[i] == y[j] else diff).append(dists[i, j])
#         ratio = np.mean(diff) / max(np.mean(same), 1e-6)
#         metrics_str = (
#             f"Sil: {sil:.3f}  DB: {db:.3f}  CH: {ch:.1f}  "
#             f"Same: {np.mean(same):.2f}  Diff: {np.mean(diff):.2f}  Ratio: {ratio:.2f}x"
#         )
#         print(f"\n  t-SNE metrics for '{title}':")
#         print(f"    Silhouette (↑): {sil:.4f}  Davies-Bouldin (↓): {db:.4f}  "
#               f"Calinski-Harabasz (↑): {ch:.1f}  Ratio: {ratio:.2f}x")

#     X_emb = TSNE(n_components=2, perplexity=min(30, len(X) - 1),
#                  learning_rate=200, max_iter=1000,
#                  random_state=42).fit_transform(X)
#     fig, ax = plt.subplots(figsize=(9, 7))
#     sc = ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap='tab20', alpha=0.7, s=20)
#     plt.colorbar(sc, ax=ax)
#     ax.set_title(f"{title}\n{metrics_str}", fontsize=9)
#     ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
#     plt.tight_layout()
#     plt.savefig(fname, dpi=120)
#     plt.show()
#     print(f"  Saved {fname}")


# def plot_training_curves(history, stopped_at):
#     epochs = range(stopped_at)
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))

#     axes[0, 0].plot(epochs, [x * 100 for x in history["train_gnn_acc"][:stopped_at]], label="Train")
#     axes[0, 0].plot(epochs, [x * 100 for x in history["test_gnn_acc"][:stopped_at]],  label="Test")
#     axes[0, 0].set_title("GNN Video Accuracy"); axes[0, 0].set_ylabel("Accuracy (%)")
#     axes[0, 0].legend()

#     axes[0, 1].plot(epochs, [x * 100 for x in history["train_cnn_acc"][:stopped_at]], label="Train")
#     axes[0, 1].plot(epochs, [x * 100 for x in history["test_cnn_acc"][:stopped_at]],  label="Test")
#     axes[0, 1].set_title("CNN Frame Accuracy"); axes[0, 1].set_ylabel("Accuracy (%)")
#     axes[0, 1].legend()

#     axes[1, 0].plot(epochs, history["train_gnn_loss"][:stopped_at], label="Train")
#     axes[1, 0].plot(epochs, history["test_gnn_loss"][:stopped_at],  label="Test")
#     axes[1, 0].set_title("GNN Loss"); axes[1, 0].set_ylabel("Loss")
#     axes[1, 0].legend()

#     axes[1, 1].plot(epochs, history["train_cnn_loss"][:stopped_at], label="Train")
#     axes[1, 1].plot(epochs, history["test_cnn_loss"][:stopped_at],  label="Test")
#     axes[1, 1].set_title("CNN Loss"); axes[1, 1].set_ylabel("Loss")
#     axes[1, 1].legend()

#     for ax in axes.flat:
#         ax.set_xlabel("Epoch")
#     plt.suptitle(f"Training vs Test — CNN+GNN  [{SESSION_NAME}]")
#     plt.tight_layout()
#     plt.savefig(spath("training_curves.png"), dpi=120)
#     plt.show()


# def pairwise_class_distances(feats, lbls, label=""):
#     if len(feats) < 2:
#         return
#     X     = np.stack(feats)
#     y     = np.array(lbls)
#     dists = pairwise_distances(X)
#     same, diff = [], []
#     for i in range(len(X)):
#         for j in range(i + 1, len(X)):
#             (same if y[i] == y[j] else diff).append(dists[i, j])
#     print(f"  {label} — same: {np.mean(same):.4f}  diff: {np.mean(diff):.4f}  "
#           f"ratio: {np.mean(diff)/max(np.mean(same),1e-6):.2f}x")


# # ── Datasets and models ──────────────────────────────────────────────────────

# base_vectors = make_sinusoidal_base_vectors(num_classes, SYNTHETIC_FEAT_DIM)

# # FIX 6: Freeze CNN backbone, only train the classifier head
# cnn_model = FrameClassifierUnifiedCNN(num_classes=num_classes).to(device)
# # for name, param in cnn_model.named_parameters():
# #     if 'classifier' not in name and 'fc' not in name:
# #         param.requires_grad = False

# # trainable_cnn = [p for p in cnn_model.parameters() if p.requires_grad]
# # print(f"CNN trainable params: {sum(p.numel() for p in trainable_cnn):,}  "
# #       f"(frozen backbone)")

# train_dataset = New_GNN_Dataset(train_files, split='train',
#                                 label_mapping=unique_label_mappings)
# test_dataset  = New_GNN_Dataset(test_files,  split='test',
#                                 label_mapping=unique_label_mappings)
# train_loader  = TorchDataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader   = TorchDataLoader(test_dataset,  batch_size=1, shuffle=False)

# if val_files:
#     val_dataset = New_GNN_Dataset(val_files, split='test',
#                                   label_mapping=unique_label_mappings)
#     val_loader  = TorchDataLoader(val_dataset, batch_size=1, shuffle=False)
#     print(f"Using separate val set ({len(val_files)} videos) for checkpointing")
# else:
#     val_loader = test_loader
#     print("No val set — using test loader for checkpoint selection")

# # Label mapping sanity check
# print("\nLabel mapping sanity check:")
# frames_s, _, lbl_s = train_dataset[0]
# expected_idx = unique_label_mappings[extract_label_from_filename(train_files[0])]
# print(f"  Video: {train_files[0]}  Dataset label: {lbl_s.item()}  "
#       f"Expected: {expected_idx}  Match: {lbl_s.item() == expected_idx}")

# gnn_model = St_GCN_Classifier(num_classes=num_classes).to(device)
# print(f"GNN params: {sum(p.numel() for p in gnn_model.parameters()):,}")

# # FIX 2: weight decay on both optimisers
# gnn_optimiser = torch.optim.Adam(gnn_model.parameters(),
#                                   lr=GNN_LR, weight_decay=GNN_WEIGHT_DECAY)
# # cnn_optimiser = torch.optim.Adam(cnn_model.parameters(), # trainable_cnn
# #                                   lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)
# cnn_optimiser = torch.optim.Adam(cnn_model.parameters(), lr=1e-4, weight_decay=0.0)
# cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimiser, T_max=NUM_EPOCHS)
# gnn_scheduler = torch.optim.lr_scheduler.StepLR(gnn_optimiser,
#                                                   step_size=GNN_LR_STEP,
#                                                   gamma=GNN_LR_GAMMA)

# train_weights = get_class_weights(train_dataset, num_classes, device)
# # FIX 3: label smoothing
# cnn_lossfn = nn.CrossEntropyLoss(weight=train_weights, label_smoothing=LABEL_SMOOTHING)
# gnn_lossfn = nn.CrossEntropyLoss(weight=train_weights, label_smoothing=LABEL_SMOOTHING)

# # ── Training ─────────────────────────────────────────────────────────────────

# history = {k: [] for k in [
#     "train_gnn_acc", "test_gnn_acc",
#     "train_cnn_acc", "test_cnn_acc",
#     "train_gnn_loss", "test_gnn_loss",
#     "train_cnn_loss", "test_cnn_loss",
# ]}

# best_val_gnn_loss = float('inf')
# best_gnn_state    = None
# best_cnn_state    = None
# early_stopper     = EarlyStopping(EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA)
# stopped_at_epoch  = NUM_EPOCHS

# tsne_cnn_feats, tsne_cnn_lbls = [], []
# tsne_gnn_feats, tsne_gnn_lbls = [], []
# cnn_class_counts = defaultdict(int)   # FIX 8: separate counters
# gnn_class_counts = defaultdict(int)

# training_start = time.time()

# for epoch in range(NUM_EPOCHS):
#     print(f"\n{'─'*45} [{SESSION_NAME}] EPOCH {epoch}/{NUM_EPOCHS-1} {'─'*5}")
#     t0 = time.time()

#     collect_tsne = (epoch == NUM_EPOCHS - 1) or early_stopper.should_stop

#     train_metrics = run_epoch(
#         train_loader, train=True,
#         tsne_cnn_feats=tsne_cnn_feats if collect_tsne else None,
#         tsne_cnn_lbls =tsne_cnn_lbls  if collect_tsne else None,
#         cnn_class_counts=cnn_class_counts if collect_tsne else defaultdict(int),
#         tsne_gnn_feats=tsne_gnn_feats if collect_tsne else None,
#         tsne_gnn_lbls =tsne_gnn_lbls  if collect_tsne else None,
#         gnn_class_counts=gnn_class_counts if collect_tsne else defaultdict(int),
#     )
#     test_metrics = run_epoch(test_loader, train=False)

#     gnn_scheduler.step()
#     cnn_scheduler.step()

#     for k, v in train_metrics.items():
#         history[f"train_{k}"].append(v)
#     for k, v in test_metrics.items():
#         history[f"test_{k}"].append(v)

#     print(f"  Train — GNN: {100*train_metrics['gnn_acc']:.1f}%  "
#           f"CNN: {100*train_metrics['cnn_acc']:.1f}%  "
#           f"GNN loss: {train_metrics['gnn_loss']:.4f}  "
#           f"CNN loss: {train_metrics['cnn_loss']:.4f}")
#     print(f"  Test  — GNN: {100*test_metrics['gnn_acc']:.1f}%  "
#           f"CNN: {100*test_metrics['cnn_acc']:.1f}%  "
#           f"GNN loss: {test_metrics['gnn_loss']:.4f}  "
#           f"CNN loss: {test_metrics['cnn_loss']:.4f}  "
#           f"({time.time()-t0:.0f}s)")

#     # FIX 7: val checkpoint — no duplicate test pass
#     if val_files:
#         val_metrics     = run_epoch(val_loader, train=False)
#         #checkpoint_loss = val_metrics['gnn_loss']
#         checkpoint_loss = val_metrics['gnn_loss'] + 0.5 * val_metrics['cnn_loss']
#         print(f"  Val   — GNN: {100*val_metrics['gnn_acc']:.1f}%  "
#               f"GNN loss: {checkpoint_loss:.4f}")
#     else:
#         checkpoint_loss = test_metrics['gnn_loss']

#     if checkpoint_loss < best_val_gnn_loss:
#         best_val_gnn_loss = checkpoint_loss
#         best_gnn_state = {k: v.clone() for k, v in gnn_model.state_dict().items()}
#         best_cnn_state = {k: v.clone() for k, v in cnn_model.state_dict().items()}
#         print(f"  → Checkpoint saved (loss {best_val_gnn_loss:.4f})")

#     early_stopper.step(checkpoint_loss)
#     if early_stopper.should_stop:
#         print(f"\n  Early stopping triggered at epoch {epoch}")
#         stopped_at_epoch = epoch + 1
#         # Collect t-SNE on the epoch we stop at
#         if not collect_tsne:
#             run_epoch(
#                 train_loader, train=False,
#                 tsne_cnn_feats=tsne_cnn_feats, tsne_cnn_lbls=tsne_cnn_lbls,
#                 cnn_class_counts=cnn_class_counts,
#                 tsne_gnn_feats=tsne_gnn_feats, tsne_gnn_lbls=tsne_gnn_lbls,
#                 gnn_class_counts=gnn_class_counts,
#             )
#         break

# print(f"\nTraining complete — {(time.time()-training_start)/60:.1f}min  "
#       f"(stopped at epoch {stopped_at_epoch})")

# # ── Restore best checkpoint ───────────────────────────────────────────────────
# if best_gnn_state is not None:
#     gnn_model.load_state_dict(best_gnn_state)
#     cnn_model.load_state_dict(best_cnn_state)
#     torch.save(best_gnn_state, spath("gnn_best.pt"))
#     torch.save(best_cnn_state, spath("cnn_best.pt"))
#     print(f"Restored best checkpoint. Saved to {OUT_DIR}/")

# # ── Final detailed test evaluation ───────────────────────────────────────────
# print("\n" + "="*60 + "\nFINAL TEST EVALUATION\n" + "="*60)

# (gnn_per_class, cnn_per_class,
#  gnn_preds, gnn_targets,
#  cnn_preds, cnn_targets,
#  final_gnn_acc) = run_test_detailed(test_loader)

# cnn_video_acc = sum(p == t for p, t in zip(cnn_preds, cnn_targets)) / max(len(cnn_preds), 1)
# print(f"\nGNN Video Accuracy: {100*final_gnn_acc:.1f}%")
# print(f"CNN Video Accuracy: {100*cnn_video_acc:.1f}%")

# print("\nPer-class accuracy (GNN | CNN):")
# for i in range(num_classes):
#     print(f"  Class {i:2d} ({get_class_name(i):12s}):  "
#           f"GNN {100*gnn_per_class[i]:.1f}%  |  CNN {100*cnn_per_class[i]:.1f}%")

# print("\nPrediction distribution (GNN):", Counter(gnn_preds))
# print("Target distribution:         ", Counter(gnn_targets))

# # ── Diagnostics ──────────────────────────────────────────────────────────────
# print("\n" + "="*60 + "\nDIAGNOSTICS\n" + "="*60)
# diagnose_zero_classes(gnn_per_class, cnn_per_class, gnn_preds, gnn_targets)

# print("\nChecking for overfitting...")
# train_detailed = run_epoch(train_loader, train=False)
# gap = train_detailed['gnn_acc'] - final_gnn_acc
# print(f"  GNN train: {100*train_detailed['gnn_acc']:.1f}%  "
#       f"test: {100*final_gnn_acc:.1f}%  gap: {100*gap:.1f}%")
# if gap > 0.3:
#     print("  ⚠ Large generalisation gap — still overfitting despite regularisation")
# else:
#     print("  ✓ Reasonable generalisation gap")

# print("\nFeature space pairwise distances:")
# pairwise_class_distances(tsne_gnn_feats, tsne_gnn_lbls, "GNN embeddings")
# pairwise_class_distances(tsne_cnn_feats, tsne_cnn_lbls, "CNN backbone features")

# # ── Plots ─────────────────────────────────────────────────────────────────────
# plot_training_curves(history, stopped_at_epoch)
# plot_gnn_vs_cnn_delta(gnn_per_class, cnn_per_class, test_dataset)

# mode_tag = "SYNTHETIC" if SYNTHETIC_MODE else "Real"
# plot_tsne(tsne_cnn_feats, tsne_cnn_lbls,
#           f"CNN backbone features [{mode_tag}]  |  {SESSION_NAME}",
#           spath("cnn_tsne_final.png"))
# plot_tsne(tsne_gnn_feats, tsne_gnn_lbls,
#           f"GNN embeddings [{mode_tag}]  |  {SESSION_NAME}",
#           spath("gnn_tsne_final.png"))

# # Loss curves
# for values, test_values, ylabel, fname in [
#     (history["train_gnn_loss"], history["test_gnn_loss"], "GNN Loss", spath("gnn_loss.png")),
#     (history["train_cnn_loss"], history["test_cnn_loss"], "CNN Loss", spath("cnn_loss.png")),
# ]:
#     _, ax = plt.subplots(1, 1)
#     ax.plot(values[:stopped_at_epoch], label="Train")
#     ax.plot(test_values[:stopped_at_epoch], label="Test")
#     ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
#     ax.set_title(f"{ylabel}  [{SESSION_NAME}]")
#     ax.legend()
#     plt.savefig(fname)
#     plt.show()

# # ── Save results ──────────────────────────────────────────────────────────────
# results_summary = {
#     "session_name":        SESSION_NAME,
#     "session_notes":       SESSION_NOTES,
#     "final_gnn_acc":       float(final_gnn_acc),
#     "final_cnn_video_acc": float(cnn_video_acc),
#     "best_val_gnn_loss":   float(best_val_gnn_loss),
#     "stopped_at_epoch":    stopped_at_epoch,
#     "train_time_min":      round((time.time() - training_start) / 60, 1),
#     "gnn_per_class_acc":   [round(float(x), 4) for x in gnn_per_class],
#     "cnn_per_class_acc":   [round(float(x), 4) for x in cnn_per_class],
#     "history":             {k: [round(float(v), 6) for v in vals]
#                             for k, vals in history.items()},
# }
# with open(spath("results.json"), "w") as fp:
#     json.dump(results_summary, fp, indent=2)

# print(f"\n{'█'*65}")
# print(f"  SESSION {SESSION_NAME} COMPLETE")
# print(f"  GNN: {100*final_gnn_acc:.1f}%   CNN: {100*cnn_video_acc:.1f}%   "
#       f"Stopped: epoch {stopped_at_epoch}   Time: {results_summary['train_time_min']:.1f}min")
# print(f"  Outputs → {OUT_DIR}/")
# print("█" * 65)

"""
CNN + GNN training script — s4-spatial-only-lambda-0.2-data-aug

Val set is carved out dynamically using StratifiedGroupKFold on the train
split (Option B). The test set is fixed and loaded from the JSON, making
results directly comparable across CNN v1, CNN v2, and this GNN script.

For each fold:
  - A fresh GNN + CNN are initialised
  - Early stopping monitors the fold's val loss
  - The best checkpoint within that fold is saved
After all folds the globally best checkpoint (lowest val loss across all
folds) is restored and evaluated once on the fixed test set.
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

SESSION_NAME  = "s3-data-aug-stratifiedkfold"
SESSION_NOTES = (
    "Dropout re-added, weight decay, label smoothing, early stopping, "
    "frozen CNN backbone, reduced GNN dim 256→128, "
    "dynamic fold-based val via StratifiedGroupKFold"
)

# ── Model / training hyperparameters ───────────────────────────────────────
GNN_LR              = 1e-3
CNN_LR              = 1e-4
NUM_EPOCHS          = 20            # per fold; early stopping cuts this short
CHUNK_SIZE          = 30
GNN_LR_STEP         = 10
GNN_LR_GAMMA        = 0.5
GNN_WEIGHT_DECAY    = 1e-3
CNN_WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING     = 0.1
GNN_DROPOUT         = 0.3
GNN_HIDDEN_DIM      = 128
INTRA_FRAME_EDGES   = True
MIN_REGION_AREA     = 50
LAMBDA              = 0.2

# ── Early stopping ──────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 5
EARLY_STOP_MIN_DELTA = 0.03

# ── Feature / graph config ─────────────────────────────────────────────────
SYNTHETIC_MODE        = False
SYNTHETIC_NOISE_SCALE = 0.1
SYNTHETIC_FEAT_DIM    = 2048
MAX_PER_CLASS_TSNE    = 70

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = f"outputs/{SESSION_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

def spath(filename):
    return os.path.join(OUT_DIR, filename)

print("\n" + "█" * 65)
print(f"  SESSION : {SESSION_NAME}")
print(f"  NOTES   : {SESSION_NOTES}")
print(f"  GNN LR={GNN_LR}  WD={GNN_WEIGHT_DECAY}  Dropout={GNN_DROPOUT}  "
      f"Hidden={GNN_HIDDEN_DIM}")
print(f"  CNN LR={CNN_LR}  WD={CNN_WEIGHT_DECAY}  LabelSmooth={LABEL_SMOOTHING}")
print(f"  Lambda={LAMBDA}  Early stop patience={EARLY_STOP_PATIENCE}  "
      f"Epochs/fold={NUM_EPOCHS}")
print(f"  Outputs → {OUT_DIR}/")
print("█" * 65 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA']   = "1"

# ── Load splits ───────────────────────────────────────────────────────────────
MIN_SAMPLES = 10
OUTPUT_FILE = f"splits_{MIN_SAMPLES}.json"

with open(OUTPUT_FILE) as fp:
    splits = json.load(fp)

train_files           = splits["train"]
test_files            = splits["test"]
unique_label_mappings = splits["label_mapping"]
num_classes           = splits["num_classes"]
K_FOLDS               = splits["k_folds"]
SEED                  = splits["seed"]
# val is intentionally empty in the JSON
assert splits.get("val", []) == [], \
    "val should be empty in JSON — val is generated dynamically here"

print(f"Train: {len(train_files)}  Test: {len(test_files)}")
print(f"Classes: {num_classes}  K_FOLDS: {K_FOLDS}  Seed: {SEED}\n")

session_config = dict(
    session_name=SESSION_NAME, session_notes=SESSION_NOTES,
    gnn_lr=GNN_LR, cnn_lr=CNN_LR, num_epochs=NUM_EPOCHS,
    gnn_weight_decay=GNN_WEIGHT_DECAY, cnn_weight_decay=CNN_WEIGHT_DECAY,
    label_smoothing=LABEL_SMOOTHING, gnn_dropout=GNN_DROPOUT,
    gnn_hidden_dim=GNN_HIDDEN_DIM, early_stop_patience=EARLY_STOP_PATIENCE,
    intra_frame_edges=INTRA_FRAME_EDGES, synthetic_mode=SYNTHETIC_MODE,
    lambda_=LAMBDA, k_folds=K_FOLDS, seed=SEED,
)
with open(spath("config.json"), "w") as fp:
    json.dump(session_config, fp, indent=2)

# ── StratifiedGroupKFold setup ────────────────────────────────────────────────
# Labels and groups built from the file list — one group per video (no leakage)
fold_labels = np.array([unique_label_mappings[extract_label_from_filename(f)]
                         for f in train_files])
fold_groups = np.array(train_files)

sgkf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

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

def get_class_weights(dataset, num_classes, device):
    counts = torch.zeros(num_classes)
    for _, _, l in dataset:
        counts[l] += 1
    w = 1.0 / (counts + 1e-6)
    return (w / w.sum() * num_classes).to(device)

def get_class_name(idx):
    return [k for k, v in unique_label_mappings.items() if v == idx][0]

base_vectors = make_sinusoidal_base_vectors(num_classes, SYNTHETIC_FEAT_DIM)

# ── GNN model ─────────────────────────────────────────────────────────────────

class St_GCN_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        H = GNN_HIDDEN_DIM
        self.node_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(SYNTHETIC_FEAT_DIM, H),
            nn.ReLU(),
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
    frame_ids     = sorted(subimages_per_frame.keys())
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

        # inter-frame edges commented out (spatial-only mode)
        # if idx < len(frame_ids) - 1:
        #     next_fid = frame_ids[idx + 1]
        #     for ci in range(len(nodes_here)):
        #         src = frame_offsets[fid] + ci
        #         for ni in range(len(subimages_per_frame[next_fid])):
        #             edges_from.append(src)
        #             edges_to.append(frame_offsets[next_fid] + ni)

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

    for i in range(0, F_len, CHUNK_SIZE):
        chunk = video_frames[i:i + CHUNK_SIZE].to(device)
        chunk = F.interpolate(chunk, size=(224, 224),
                              mode='bilinear', align_corners=False)

        if not SYNTHETIC_MODE:
            logits, feature_maps = cnn_model(chunk)
            all_logits.append(logits)
            feat_vec = feature_maps.mean(dim=(2, 3))
            all_feat_vecs.append(feat_vec.detach())
            video_labels_chunk = video_label.repeat(chunk.size(0))
            allcnnlosses = allcnnlosses + cnn_lossfn(logits, video_labels_chunk)
        else:
            logits, feature_maps = cnn_model(chunk)
            allcnnlosses = torch.tensor(0.0, requires_grad=True, device=device)

        for j in range(chunk.size(0)):
            frame_idx = i + j
            subimages_per_frame[global_frame_count] = []

            if SYNTHETIC_MODE:
                node_feat = synthetic_node_feature(
                    int(video_label.item()), base_vectors,
                    SYNTHETIC_NOISE_SCALE, device)
                subimages_per_frame[global_frame_count].append(node_feat)
                global_frame_count += 1
            else:
                mask    = video_masks[frame_idx].float()
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

    return graph, avg_cnn_loss, all_logits, backbone_feat_vec

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
        self.best_loss   = float('inf')
        self.counter     = 0
        self.should_stop = False

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

                graph, avg_cnn_loss, all_logits, backbone_feat = process_video_to_graph(
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

                total_vids   += 1
                correct_gnn  += (gnn_pred.argmax(1) == video_label.unsqueeze(0)).float().sum().item()
                sum_gnn_loss += gnn_loss.item()
                sum_cnn_loss += avg_cnn_loss.item() if not SYNTHETIC_MODE else 0.0

                if all_logits:
                    all_probs    = F.softmax(torch.cat(all_logits, dim=0), dim=1)
                    avg_prob     = all_probs.mean(dim=0)
                    cnn_pred_vid = avg_prob.argmax().item()
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
        "gnn_acc":  correct_gnn / n,
        "cnn_acc":  correct_cnn_vids / n,
        "gnn_loss": sum_gnn_loss / n,
        "cnn_loss": sum_cnn_loss / n,
    }

# ── Full test with per-class breakdown ─────────────────────────────────────────

def run_test_detailed(loader):
    gnn_model.eval(); cnn_model.eval()
    gnn_preds, gnn_targets = [], []
    cnn_preds, cnn_targets = [], []
    correct_vids = total_vids = 0

    with torch.no_grad():
        for f, m, l in loader:
            for vid_idx in range(f.size(0)):
                video_frames = f[vid_idx]
                video_label  = l[vid_idx].to(device)
                video_masks  = m[vid_idx]

                graph, _, all_logits, _ = process_video_to_graph(
                    video_frames, video_masks, video_label,
                    cnn_model, base_vectors, device)

                total_vids += 1

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
    return gnn_per_class, cnn_per_class, gnn_preds, gnn_targets, cnn_preds, cnn_targets, gnn_acc

# ── Diagnostics ────────────────────────────────────────────────────────────────

def diagnose_zero_classes(gnn_per_class, cnn_per_class, gnn_preds, gnn_targets):
    zero_classes = [i for i, acc in enumerate(gnn_per_class) if acc == 0.0]
    print(f"\n{'─'*60}")
    print(f"GNN zero-accuracy classes ({len(zero_classes)}): {zero_classes}")
    for cls in zero_classes:
        wrong     = [gnn_preds[i] for i in range(len(gnn_targets)) if gnn_targets[i] == cls]
        count     = sum(1 for t in gnn_targets if t == cls)
        name      = get_class_name(cls)
        top       = Counter(wrong).most_common(3)
        top_named = [(get_class_name(p), c) for p, c in top]
        print(f"  Class {cls:2d} ({name:12s}) | {count} test vids | "
              f"predicted as: {top_named}")
    both_zero        = [i for i in range(num_classes)
                        if gnn_per_class[i] == 0.0 and cnn_per_class[i] == 0.0]
    cnn_good_gnn_bad = [i for i in range(num_classes)
                        if gnn_per_class[i] == 0.0 and cnn_per_class[i] > 0.3]
    print(f"\nClasses 0% in BOTH: {both_zero}")
    print(f"CNN>30% but GNN 0%: {[(c, get_class_name(c)) for c in cnn_good_gnn_bad]}")


def plot_gnn_vs_cnn_delta(gnn_per_class, cnn_per_class, test_dataset):
    counts     = Counter(int(l) for _, _, l in test_dataset)
    sizes      = np.array([counts[i] for i in range(num_classes)])
    heights    = 0.3 + (sizes / max(sizes)) * 0.7
    delta      = (gnn_per_class - cnn_per_class) * 100
    sorted_idx = np.argsort(delta)
    delta_s    = delta[sorted_idx]
    heights_s  = heights[sorted_idx]
    names_s    = [get_class_name(i) for i in sorted_idx]
    colors     = ['green' if d > 0 else 'steelblue' for d in delta_s]

    plt.figure(figsize=(9, 11))
    y_pos = np.arange(len(delta_s))
    plt.barh(y_pos, delta_s, height=heights_s, color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    max_abs = max(np.max(np.abs(delta_s)), 1)
    plt.xlim(-max_abs * 1.15, max_abs * 1.15)
    plt.yticks(y_pos, names_s, fontsize=8)
    plt.xlabel("Δ Accuracy (%) (GNN − CNN)")
    plt.title(f"GNN vs CNN per Class  [{SESSION_NAME}]\n(bar thickness = test set size)")
    for i, v in enumerate(delta_s):
        plt.text(v + (0.5 if v >= 0 else -3), i, f"{v:.1f}", va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(spath("gnn_vs_cnn_delta.png"), dpi=120)
    plt.show()


def plot_tsne(feats, lbls, title, fname):
    if len(feats) < 5:
        print(f"  t-SNE skipped — not enough samples ({len(feats)})")
        return
    X = np.stack(feats)
    y = np.array(lbls)
    metrics_str = ""
    if len(np.unique(y)) >= 2:
        sil   = silhouette_score(X, y, metric='euclidean', sample_size=min(2000, len(X)))
        db    = davies_bouldin_score(X, y)
        ch    = calinski_harabasz_score(X, y)
        dists = pairwise_distances(X)
        same, diff = [], []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                (same if y[i] == y[j] else diff).append(dists[i, j])
        ratio = np.mean(diff) / max(np.mean(same), 1e-6)
        metrics_str = (
            f"Sil: {sil:.3f}  DB: {db:.3f}  CH: {ch:.1f}  "
            f"Same: {np.mean(same):.2f}  Diff: {np.mean(diff):.2f}  Ratio: {ratio:.2f}x"
        )
        print(f"\n  t-SNE metrics '{title}':")
        print(f"    Sil={sil:.4f}  DB={db:.4f}  CH={ch:.1f}  Ratio={ratio:.2f}x")
    X_emb = TSNE(n_components=2, perplexity=min(30, len(X) - 1),
                 learning_rate=200, max_iter=1000,
                 random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap='tab20', alpha=0.7, s=20)
    plt.colorbar(sc, ax=ax)
    ax.set_title(f"{title}\n{metrics_str}", fontsize=9)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(fname, dpi=120); plt.show()
    print(f"  Saved {fname}")


def plot_training_curves(history, stopped_at, fold_idx):
    epochs = range(stopped_at)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [x*100 for x in history["train_gnn_acc"][:stopped_at]], label="Train")
    axes[0, 0].plot(epochs, [x*100 for x in history["val_gnn_acc"][:stopped_at]],   label="Val")
    axes[0, 0].set_title("GNN Video Accuracy"); axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [x*100 for x in history["train_cnn_acc"][:stopped_at]], label="Train")
    axes[0, 1].plot(epochs, [x*100 for x in history["val_cnn_acc"][:stopped_at]],   label="Val")
    axes[0, 1].set_title("CNN Frame Accuracy"); axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, history["train_gnn_loss"][:stopped_at], label="Train")
    axes[1, 0].plot(epochs, history["val_gnn_loss"][:stopped_at],   label="Val")
    axes[1, 0].set_title("GNN Loss"); axes[1, 0].set_ylabel("Loss"); axes[1, 0].legend()
    axes[1, 1].plot(epochs, history["train_cnn_loss"][:stopped_at], label="Train")
    axes[1, 1].plot(epochs, history["val_cnn_loss"][:stopped_at],   label="Val")
    axes[1, 1].set_title("CNN Loss"); axes[1, 1].set_ylabel("Loss"); axes[1, 1].legend()
    for ax in axes.flat: ax.set_xlabel("Epoch")
    plt.suptitle(f"Training vs Val — CNN+GNN  [{SESSION_NAME}]  Fold {fold_idx+1}")
    plt.tight_layout()
    plt.savefig(spath(f"training_curves_fold{fold_idx+1}.png"), dpi=120)
    plt.show()


def pairwise_class_distances(feats, lbls, lbl=""):
    if len(feats) < 2: return
    X = np.stack(feats); y = np.array(lbls)
    dists = pairwise_distances(X)
    same, diff = [], []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            (same if y[i] == y[j] else diff).append(dists[i, j])
    print(f"  {lbl} — same: {np.mean(same):.4f}  diff: {np.mean(diff):.4f}  "
          f"ratio: {np.mean(diff)/max(np.mean(same),1e-6):.2f}x")

# ── Fixed test dataset (shared across all folds) ───────────────────────────────
test_dataset = New_GNN_Dataset(test_files, split='test',
                               label_mapping=unique_label_mappings)
test_loader  = TorchDataLoader(test_dataset, batch_size=1, shuffle=False)

# ── Global best tracking across all folds ─────────────────────────────────────
global_best_val_loss = float('inf')
global_best_gnn_state = None
global_best_cnn_state = None
fold_summaries        = []

tsne_cnn_feats, tsne_cnn_lbls = [], []
tsne_gnn_feats, tsne_gnn_lbls = [], []
cnn_class_counts = defaultdict(int)
gnn_class_counts = defaultdict(int)

training_start = time.time()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         FOLD TRAINING LOOP                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

for fold_idx, (tr_idx, val_idx) in enumerate(
    sgkf.split(np.zeros(len(fold_labels)), fold_labels, fold_groups)
):
    fold_train_files = [train_files[i] for i in tr_idx]
    fold_val_files   = [train_files[i] for i in val_idx]

    print(f"\n{'█'*65}")
    print(f"  FOLD {fold_idx+1}/{K_FOLDS}  "
          f"mini-train={len(fold_train_files)}  val={len(fold_val_files)}")
    print(f"{'█'*65}")

    # ── Datasets for this fold ────────────────────────────────────────────
    fold_train_dataset = New_GNN_Dataset(fold_train_files, split='train',
                                         label_mapping=unique_label_mappings)
    fold_val_dataset   = New_GNN_Dataset(fold_val_files,   split='test',
                                         label_mapping=unique_label_mappings)
    fold_train_loader  = TorchDataLoader(fold_train_dataset, batch_size=1, shuffle=True)
    fold_val_loader    = TorchDataLoader(fold_val_dataset,   batch_size=1, shuffle=False)

    # ── Fresh models for each fold ────────────────────────────────────────
    cnn_model = FrameClassifierUnifiedCNN(num_classes=num_classes).to(device)
    for name, param in cnn_model.named_parameters():
        if 'classifier' not in name and 'fc' not in name:
            param.requires_grad = False
    trainable_cnn = [p for p in cnn_model.parameters() if p.requires_grad]
    print(f"  CNN trainable params: {sum(p.numel() for p in trainable_cnn):,} "
          f"(backbone frozen)")

    gnn_model = St_GCN_Classifier(num_classes=num_classes).to(device)

    # ── Loss functions — weights from fold train set ──────────────────────
    train_weights = get_class_weights(fold_train_dataset, num_classes, device)
    cnn_lossfn    = nn.CrossEntropyLoss(weight=train_weights,
                                        label_smoothing=LABEL_SMOOTHING)
    gnn_lossfn    = nn.CrossEntropyLoss(weight=train_weights,
                                        label_smoothing=LABEL_SMOOTHING)

    gnn_optimiser = torch.optim.Adam(gnn_model.parameters(),
                                      lr=GNN_LR, weight_decay=GNN_WEIGHT_DECAY)
    cnn_optimiser = torch.optim.Adam(trainable_cnn,
                                      lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)
    gnn_scheduler = torch.optim.lr_scheduler.StepLR(
        gnn_optimiser, step_size=GNN_LR_STEP, gamma=GNN_LR_GAMMA)
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimiser, T_max=NUM_EPOCHS)

    # ── Per-fold history and early stopping ───────────────────────────────
    history = {k: [] for k in [
        "train_gnn_acc", "val_gnn_acc",
        "train_cnn_acc", "val_cnn_acc",
        "train_gnn_loss", "val_gnn_loss",
        "train_cnn_loss", "val_cnn_loss",
    ]}

    early_stopper    = EarlyStopping(EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA)
    best_fold_loss   = float('inf')
    best_fold_gnn    = None
    best_fold_cnn    = None
    stopped_at_epoch = NUM_EPOCHS
    is_last_fold     = (fold_idx == K_FOLDS - 1)

    for epoch in range(NUM_EPOCHS):
        print(f"\n  {'─'*40} FOLD {fold_idx+1} EPOCH {epoch}/{NUM_EPOCHS-1}")
        t0 = time.time()

        # Collect t-SNE on last epoch of last fold, or when stopping early
        collect_tsne = is_last_fold and (
            epoch == NUM_EPOCHS - 1 or early_stopper.should_stop
        )

        train_metrics = run_epoch(
            fold_train_loader, train=True,
            tsne_cnn_feats=tsne_cnn_feats if collect_tsne else None,
            tsne_cnn_lbls =tsne_cnn_lbls  if collect_tsne else None,
            cnn_class_counts=cnn_class_counts if collect_tsne else defaultdict(int),
            tsne_gnn_feats=tsne_gnn_feats if collect_tsne else None,
            tsne_gnn_lbls =tsne_gnn_lbls  if collect_tsne else None,
            gnn_class_counts=gnn_class_counts if collect_tsne else defaultdict(int),
        )
        val_metrics = run_epoch(fold_val_loader, train=False)

        gnn_scheduler.step()
        cnn_scheduler.step()

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)

        # Checkpoint loss: combined, same formula as total_loss
        checkpoint_loss = val_metrics['gnn_loss'] + 0.5 * val_metrics['cnn_loss']

        print(f"  Train — GNN: {100*train_metrics['gnn_acc']:.1f}%  "
              f"CNN: {100*train_metrics['cnn_acc']:.1f}%  "
              f"GNN loss: {train_metrics['gnn_loss']:.4f}  "
              f"CNN loss: {train_metrics['cnn_loss']:.4f}")
        print(f"  Val   — GNN: {100*val_metrics['gnn_acc']:.1f}%  "
              f"CNN: {100*val_metrics['cnn_acc']:.1f}%  "
              f"checkpoint loss: {checkpoint_loss:.4f}  "
              f"({time.time()-t0:.0f}s)")

        if checkpoint_loss < best_fold_loss:
            best_fold_loss = checkpoint_loss
            best_fold_gnn  = {k: v.clone() for k, v in gnn_model.state_dict().items()}
            best_fold_cnn  = {k: v.clone() for k, v in cnn_model.state_dict().items()}
            print(f"  → Fold checkpoint saved (loss {best_fold_loss:.4f})")

        early_stopper.step(checkpoint_loss)
        if early_stopper.should_stop:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            stopped_at_epoch = epoch + 1
            # Collect t-SNE if last fold and not already collected
            if is_last_fold and not collect_tsne:
                run_epoch(
                    fold_train_loader, train=False,
                    tsne_cnn_feats=tsne_cnn_feats, tsne_cnn_lbls=tsne_cnn_lbls,
                    cnn_class_counts=cnn_class_counts,
                    tsne_gnn_feats=tsne_gnn_feats, tsne_gnn_lbls=tsne_gnn_lbls,
                    gnn_class_counts=gnn_class_counts,
                )
            break

    # ── Track global best across folds ────────────────────────────────────
    if best_fold_loss < global_best_val_loss:
        global_best_val_loss  = best_fold_loss
        global_best_gnn_state = best_fold_gnn
        global_best_cnn_state = best_fold_cnn
        print(f"\n  ★ New global best from Fold {fold_idx+1} "
              f"(val loss {global_best_val_loss:.4f})")

    plot_training_curves(history, stopped_at_epoch, fold_idx)
    fold_summaries.append({
        "fold": fold_idx + 1,
        "best_val_loss":  round(best_fold_loss, 4),
        "stopped_epoch":  stopped_at_epoch,
        "val_gnn_acc":    round(val_metrics['gnn_acc'], 4),
        "val_cnn_acc":    round(val_metrics['cnn_acc'], 4),
    })

print(f"\nAll folds complete — {(time.time()-training_start)/60:.1f}min total")
print("\nFold summary:")
for s in fold_summaries:
    print(f"  Fold {s['fold']}: val_loss={s['best_val_loss']:.4f}  "
          f"GNN={100*s['val_gnn_acc']:.1f}%  CNN={100*s['val_cnn_acc']:.1f}%  "
          f"stopped@ep{s['stopped_epoch']}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   RESTORE BEST & EVALUATE ON TEST                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Re-instantiate models with correct architecture before loading state
cnn_model = FrameClassifierUnifiedCNN(num_classes=num_classes).to(device)
gnn_model = St_GCN_Classifier(num_classes=num_classes).to(device)

# Rebuild loss functions using full training set weights for test evaluation
full_train_dataset = New_GNN_Dataset(train_files, split='train',
                                     label_mapping=unique_label_mappings)
train_weights  = get_class_weights(full_train_dataset, num_classes, device)
cnn_lossfn     = nn.CrossEntropyLoss(weight=train_weights, label_smoothing=LABEL_SMOOTHING)
gnn_lossfn     = nn.CrossEntropyLoss(weight=train_weights, label_smoothing=LABEL_SMOOTHING)

gnn_model.load_state_dict(global_best_gnn_state)
cnn_model.load_state_dict(global_best_cnn_state)
torch.save(global_best_gnn_state, spath("gnn_best.pt"))
torch.save(global_best_cnn_state, spath("cnn_best.pt"))
print(f"\nRestored global best checkpoint (val loss {global_best_val_loss:.4f})")

print("\n" + "="*60 + "\nFINAL TEST EVALUATION\n" + "="*60)

(gnn_per_class, cnn_per_class,
 gnn_preds, gnn_targets,
 cnn_preds, cnn_targets,
 final_gnn_acc) = run_test_detailed(test_loader)

cnn_video_acc = sum(p == t for p, t in zip(cnn_preds, cnn_targets)) / max(len(cnn_preds), 1)
print(f"\nGNN Video Accuracy: {100*final_gnn_acc:.1f}%")
print(f"CNN Video Accuracy: {100*cnn_video_acc:.1f}%")

print("\nPer-class accuracy (GNN | CNN):")
for i in range(num_classes):
    print(f"  Class {i:2d} ({get_class_name(i):12s}):  "
          f"GNN {100*gnn_per_class[i]:.1f}%  |  CNN {100*cnn_per_class[i]:.1f}%")

print("\nPrediction distribution (GNN):", Counter(gnn_preds))
print("Target distribution:         ", Counter(gnn_targets))

print("\n" + "="*60 + "\nDIAGNOSTICS\n" + "="*60)
diagnose_zero_classes(gnn_per_class, cnn_per_class, gnn_preds, gnn_targets)

print("\nChecking for overfitting (full train set vs test)...")
full_train_loader  = TorchDataLoader(full_train_dataset, batch_size=1, shuffle=False)
train_detailed     = run_epoch(full_train_loader, train=False)
gap = train_detailed['gnn_acc'] - final_gnn_acc
print(f"  GNN train: {100*train_detailed['gnn_acc']:.1f}%  "
      f"test: {100*final_gnn_acc:.1f}%  gap: {100*gap:.1f}%")
print("  ⚠ Large generalisation gap" if gap > 0.3 else "  ✓ Reasonable generalisation gap")

print("\nFeature space pairwise distances:")
pairwise_class_distances(tsne_gnn_feats, tsne_gnn_lbls, "GNN embeddings")
pairwise_class_distances(tsne_cnn_feats, tsne_cnn_lbls, "CNN backbone features")

# ── Plots ──────────────────────────────────────────────────────────────────────
plot_gnn_vs_cnn_delta(gnn_per_class, cnn_per_class, test_dataset)

mode_tag = "SYNTHETIC" if SYNTHETIC_MODE else "Real"
plot_tsne(tsne_cnn_feats, tsne_cnn_lbls,
          f"CNN backbone features [{mode_tag}]  |  {SESSION_NAME}",
          spath("cnn_tsne_final.png"))
plot_tsne(tsne_gnn_feats, tsne_gnn_lbls,
          f"GNN embeddings [{mode_tag}]  |  {SESSION_NAME}",
          spath("gnn_tsne_final.png"))

for values, val_values, ylabel, fname in [
    (history["train_gnn_loss"], history["val_gnn_loss"],
     "GNN Loss (last fold)", spath("gnn_loss.png")),
    (history["train_cnn_loss"], history["val_cnn_loss"],
     "CNN Loss (last fold)", spath("cnn_loss.png")),
]:
    _, ax = plt.subplots(1, 1)
    ax.plot(values[:stopped_at_epoch],     label="Train")
    ax.plot(val_values[:stopped_at_epoch], label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  [{SESSION_NAME}]"); ax.legend()
    plt.savefig(fname); plt.show()

# ── Save results ───────────────────────────────────────────────────────────────
results_summary = {
    "session_name":         SESSION_NAME,
    "session_notes":        SESSION_NOTES,
    "final_gnn_acc":        float(final_gnn_acc),
    "final_cnn_video_acc":  float(cnn_video_acc),
    "global_best_val_loss": float(global_best_val_loss),
    "train_time_min":       round((time.time() - training_start) / 60, 1),
    "fold_summaries":       fold_summaries,
    "gnn_per_class_acc":    [round(float(x), 4) for x in gnn_per_class],
    "cnn_per_class_acc":    [round(float(x), 4) for x in cnn_per_class],
}
with open(spath("results.json"), "w") as fp:
    json.dump(results_summary, fp, indent=2)

print(f"\n{'█'*65}")
print(f"  SESSION {SESSION_NAME} COMPLETE")
print(f"  GNN: {100*final_gnn_acc:.1f}%   CNN: {100*cnn_video_acc:.1f}%   "
      f"Time: {results_summary['train_time_min']:.1f}min")
print(f"  Outputs → {OUT_DIR}/")
print("█" * 65)