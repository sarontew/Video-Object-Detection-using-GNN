import os
import time
import random
import cv2
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import torch_geometric
from torch import Tensor
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as torchfunc
from collections import defaultdict
from utils import save_region_features, get_object_label, get_file_names, extract_label_from_filename
from torchvision import transforms, models
from torchvision.models import resnet50
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from cnn_baseline import FrameClassifier
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from skimage.measure import label, regionprops
from gnn_dataset import New_GNN_Dataset
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models.detection.roi_heads import roi_align

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_features(x, title):
    x_np = x.detach().cpu().numpy()
    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(x_np)

    plt.scatter(x_2d[:,0], x_2d[:,1], s=20)
    plt.title(title)
    plt.show()


def evaluate(cnn_model, graphs_test_loader, epoch=0):
    # ------ Testing using CNN classifier
    print("Testing using CNN classifier")
    test_start_time = time.time()
    cnn_model.eval()
    test_loss = 0
    correct_frames = 0  # frame-level accuracy
    video_probs_dict = {}  # accumulate frame probs per video
    all_preds = []
    all_targets = []
    allframes = 0
    testing_cnn_losses_plot = []

    with torch.no_grad():
        for f, r, l in graphs_test_loader:
            f = f.float() / 255.0          # convert uint8 -> float32 and normalize
            f = f.permute(0, 1, 4, 2, 3)  # (B, F, H, W, C) -> (B, F, C, H, W)
            
            for vid_idx in range(f.size(0)):
                allframes += f.size(1)
                video_frames = f[vid_idx]           # (F, H, W, C)
                video_label = l[vid_idx]            # scalar class for this video
                #videoregions = r[vid_idx]

                frames = video_frames.to(device)
                video_label = video_label.to(device)

                video_labels = video_label.repeat(video_frames.size(0))

                pred, _ = cnn_model(frames)  # (batch_size, num_classes)
                # print("pred argmax", pred.argmax(1))
                # print("target", target)
                test_cnn_loss = cnn_lossfn(pred, video_labels).item()
                test_loss += test_cnn_loss
                testing_cnn_losses_plot.append(test_cnn_loss)

                # predicted class
                _, predicted = torch.max(pred, 1)

                # for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(video_labels.cpu().numpy())
                
                # frame-level accuracy
                # print("gnn pred argmax", pred.argmax(1))
                # print("video labels", video_labels)
                correct_frames += (pred.argmax(1) == video_labels).type(torch.float).sum().item()
                
                # softmax probabilities
                probs = torchfunc.softmax(pred, dim=1)
                
                # accumulate per video (using target as video ID)
                for i, vid in enumerate(video_labels):
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

    size = len(graphs_test_loader.dataset) # num of videos
    num_videos = len(video_probs_dict)
    print("correct frames is", correct_frames)
    print("size is", size)
    print("all frames is", allframes)
    # frame_acc = correct_frames / size
    frame_acc = correct_frames / allframes
    video_acc = correct_videos / num_videos
    test_loss /= len(graphs_test_loader)

    print(f"Frame Accuracy: {100*frame_acc:.1f}%, Video Accuracy: {100*video_acc:.1f}%, Loss: {test_loss:.6f}")
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print(f"Test time is {test_time}")

    cm = confusion_matrix(all_targets, all_preds)
    # print("Confusion Matrix:")
    # print(cm)

    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    index = str(epoch) if epoch!= 0 else ""

    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"Class {i}: {acc*100:.2f}%")

    _, ax = plt.subplots(1,1)
    ax.set_xlabel('Iter')
    ax.set_ylabel('Test CNN loss')
    ax.set_title(f"Test CNN loss {index} training")
    # Raw loss (optional, can make it lighter)
    ax.plot(testing_cnn_losses_plot, alpha=0.3, label="Raw cnn Loss")
    # Smoothed loss
    smoothed_cnn_test = moving_average(testing_cnn_losses_plot, window=20)
    ax.plot(range(len(smoothed_cnn_test)), smoothed_cnn_test, label="Smoothed CNN Loss", linewidth=2)
    ax.legend()
    plt.savefig(f"Test_CNN{index}.png")
    plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
class St_GCN_Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.hidden_channels = 256  

        self.node_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # stabilises varying graph sizes
        )

        self.conv1 = GCNConv(256, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)

        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_proj(x)       # (N, 256)

        x = self.conv1(x, edge_index)
        x = torchfunc.relu(x)
        x = self.conv2(x, edge_index)  # FIX 1: actually use conv2
        x = torchfunc.relu(x)

        # FIX 3: normalise pooled graph repr to reduce scale variance between graphs
        #pooled = global_mean_pool(x, batch)
        pooled = global_max_pool(x, batch)

        #pooled = torchfunc.normalize(pooled, dim=1)

        return self.classifier(pooled)

training_start_time = time.time()

all_valid_file_names = get_file_names('train.txt') + get_file_names('val.txt')
#all_valid_file_names = all_valid_file_names[0:2]

#all_valid_file_names = ['1186_cut_chilli', '4176_cut_cloth', '4174_cut_cloth', '4331_cut_cloth', '4320_tear_dough', '226_squeeze_dough', '2218_empty_raisin', '455_fold_box',  '1184_cut_chilli']
#all_valid_file_names = ['455_fold_box', '4320_tear_dough', '226_squeeze_dough', '3705_flatten_box', '1186_cut_chilli', '4176_cut_cloth', '4174_cut_cloth',]
#all_valid_file_names = ['574_cut_peach']
# train_files = all_valid_file_names
# test_files = all_valid_file_names

from collections import defaultdict
import random

MIN_SAMPLES = 5
MAX_SAMPLES = 50
TEST_SPLIT = 0.2

class_to_files = defaultdict(list)

# Build mapping
for file in all_valid_file_names:
    cl = get_object_label(file)
    class_to_files[cl].append(file)

print("Before filtering:", len(class_to_files), "classes")

# removing small classes
filtered_class_to_files = {}
for cls, files in class_to_files.items():
    if len(files) >= MIN_SAMPLES:
        filtered_class_to_files[cls] = files

print("After filtering:", len(filtered_class_to_files), "classes")

# cap large classes
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
all_labels = sorted({extract_label_from_filename(f) for f in all_files})
print("all labels", all_labels)


# Create consistent label → ID mapping
unique_label_mappings = {label: idx for idx, label in enumerate(all_labels)}
num_classes = len(unique_label_mappings)
print("num_classes", num_classes)
print("unique_label_mappings", unique_label_mappings)

cnn_model = FrameClassifier(num_classes=num_classes).to(device) # TODO: fix num_classes, get unique object/action labels based on videos
print("cnn model created")
train_video_dataset = New_GNN_Dataset(video_names=train_files, split='train')
print("train_video_dataset done")
graphs_train_loader = TorchDataLoader(train_video_dataset, batch_size=1, shuffle=True)
test_video_dataset = New_GNN_Dataset(video_names=test_files, split='test')
print("test_video_dataset done")
graphs_test_loader = TorchDataLoader(test_video_dataset, batch_size=1, shuffle=False)

gnn_model = St_GCN_Classifier(num_classes=num_classes).to(device)
# # After creating cnn_model, freeze the backbone
# for param in cnn_model.backbone.parameters():
#     param.requires_grad = False

# Only train the classifier head of CNN + full GNN
gnn_optimiser = torch.optim.Adam(gnn_model.parameters(), lr=0.001) ## too large
cnn_optimiser = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

## sgd
cnn_lossfn = torch.nn.CrossEntropyLoss()
gnn_lossfn = torch.nn.CrossEntropyLoss()

seen_labels = []
total_losses_for_plot = []
gnn_losses_for_plot = []
cnn_losses_for_plot = []

gnn_scheduler = torch.optim.lr_scheduler.StepLR(
    gnn_optimiser, step_size=5, gamma=0.5
)

loss_spiking_classes_per_epoch = {} # epoch: loss spiking classes


for epoch in range(20):
    print("-------------EPOCH--------------", epoch)
    gnn_model.train()
    cnn_model.train()

    total_loss = 0
    total_gnn_loss = 0
    num_vids = 0

    loss_spiking_classes_per_epoch[epoch] = []
    

    for f, m, l in graphs_train_loader:
        f = f.float() / 255.0
        f = f.permute(0, 1, 4, 2, 3)  # (B, F, C, H, W)

        gnn_optimiser.zero_grad()
        cnn_optimiser.zero_grad()

        for vid_idx in range(f.size(0)):
            video_frames = f[vid_idx]
            video_label = l[vid_idx].to(device)
            video_masks = m[vid_idx]

            # print("video_label value:", video_label.item())
            # print("num_classes:", num_classes)
            # assert video_label.item() < num_classes, "Label out of range!"

            chunk_size = 30
            F_len = video_frames.shape[0]

            # Initialise once per vid
            subimages_per_frame = {}
            global_frame_count = 0
            allcnnlosses = 0.0
            # make sure init edges once per video, outside all inner loops
            edges_connecting_from = []
            edges_connecting_to = []

            for i in range(0, F_len, chunk_size):
                chunk = video_frames[i:i+chunk_size].to(device)
                chunk = torchfunc.interpolate(chunk, size=(224, 224),
                                            mode='bilinear', align_corners=False)

                outputs, feature_maps = cnn_model(chunk)

                video_labels_chunk = video_label.repeat(chunk.size(0))
                loss = cnn_lossfn(outputs, video_labels_chunk)
                if loss.item() > 3:
                    #print("spiking for epoch gonna append label", epoch, video_label)
                    loss_spiking_classes_per_epoch[epoch].append(video_label)
                allcnnlosses = allcnnlosses + loss

                for j in range(chunk.size(0)):
                    frame_idx = i + j
                    subimages_per_frame[global_frame_count] = []

                    mask = video_masks[frame_idx].float()
                    mask_np = mask.cpu().numpy().astype("uint8")
                    labeled = label(mask_np, connectivity=2)
                    regions = regionprops(labeled)

                    # Mask at (28,28) to match backbone feature spatial size
                    mask_resized = torchfunc.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(28, 28), mode='nearest'
                    ).squeeze()  # (28, 28)

                    backbone_features = feature_maps[j].unsqueeze(0)  # (1, 512, 28, 28)

                    orig_h, orig_w = video_frames[frame_idx].shape[1:]
                    scale_x = 28 / orig_w
                    scale_y = 28 / orig_h

                    if len(regions) == 0:
                        # fallback node so frame is never empty,, not sure
                        masked_feat = backbone_features.squeeze(0) * mask_resized.to(device)  # (512, 28, 28)
                        masked_feat = torchfunc.adaptive_avg_pool2d(masked_feat, (7, 7))       # (512, 7, 7)
                        subimages_per_frame[global_frame_count].append(masked_feat)
                    else:
                        for rs in regions:
                        # MAX_REGIONS = 5
                        # for rs in regions[:MAX_REGIONS]: 
                            min_row, min_col, max_row, max_col = rs.bbox
                            if (max_row - min_row) < 1 or (max_col - min_col) < 1:
                                continue

                            bbox = torch.tensor([[
                                0,
                                min_col * scale_x,
                                min_row * scale_y,
                                max_col * scale_x,
                                max_row * scale_y
                            ]], dtype=torch.float32).to(device)

                            feat_map = backbone_features.squeeze(0).detach().cpu()  # [1024,14,14]
                            mean_map = feat_map.mean(dim=0)
                            masked_feat = feat_map * mask_resized.cpu()  # (1024,28,28)
                            # # Sum or mean only over masked region
                            mean_map = masked_feat.mean(dim=0)
                            if np.average(mean_map) == 0 or np.average(mean_map) <= 1e-3:
                                #print("0 or low mean map skipping chunck", np.average(mean_map))
                                continue
                            # feature_grid = np.zeros((28, 28))
                            # x1 = int(min_col * scale_x)
                            # y1 = int(min_row * scale_y)
                            # x2 = int(max_col * scale_x)
                            # y2 = int(max_row * scale_y)
                            # feature_grid[y1:y2, x1:x2] = 1
                            # plt.imshow(feature_grid, cmap='jet')
                            # plt.title("BBox on Feature Map (14x14)")
                            # plt.colorbar()
                            # plt.show()

                            # output_size=(7,7) or (28,28)
                            pooled = roi_align(
                                input=backbone_features,
                                boxes=bbox,
                                output_size=(7, 7)
                            ).squeeze(0)  # (512, 7, 7)
                            subimages_per_frame[global_frame_count].append(pooled)

                        # degenerate bbox fallback-- double check
                        if len(subimages_per_frame[global_frame_count]) == 0:
                            masked_feat = backbone_features.squeeze(0) * mask_resized.to(device)
                            masked_feat = torchfunc.adaptive_avg_pool2d(masked_feat, (7, 7))
                            subimages_per_frame[global_frame_count].append(masked_feat)

                    global_frame_count += 1
                # end of frame loop
            # end of chunk loop

            # graph after all chunks/frames processed
            frame_ids = sorted(subimages_per_frame.keys())
            frame_offsets = {}
            offset = 0
            for fid in frame_ids:
                frame_offsets[fid] = offset
                offset += len(subimages_per_frame[fid])

            all_nodes = []
            for idx, fid in enumerate(frame_ids):
                nodes_this_frame = subimages_per_frame[fid]
                all_nodes.extend(nodes_this_frame)

                if idx == len(frame_ids) - 1:
                    continue

                next_fid = frame_ids[idx + 1]
                for ci in range(len(nodes_this_frame)):
                    src = frame_offsets[fid] + ci
                    for ni in range(len(subimages_per_frame[next_fid])):
                        tgt = frame_offsets[next_fid] + ni
                        edges_connecting_from.append(src)
                        edges_connecting_to.append(tgt)

            if len(all_nodes) == 0:
                print(f"Skipping video {vid_idx} — no nodes")
                continue

            # print("Num nodes:", len(all_nodes))
            # print("Num edges:", len(edges_connecting_from))
            # print("Nodes per frame:", {fid: len(subimages_per_frame[fid]) for fid in frame_ids})

            nodes_tensor = torch.stack(all_nodes)  # (N, 512, 7, 7) — all same shape
            #print("nodes_tensor requires_grad:", nodes_tensor.requires_grad)
            # print("nodes_tensor grad_fn:", nodes_tensor.grad_fn)
            edge_index = torch.tensor(
                [edges_connecting_from, edges_connecting_to], dtype=torch.long
            ).to(device)

            # print("Node feature std:", nodes_tensor.std().item())
            # print("Node feature mean:", nodes_tensor.mean().item())
            x_flat = nodes_tensor.mean(dim=(2,3))  # (N, 512)
            pairwise = torch.cdist(x_flat, x_flat)
            #print("Mean pairwise node distance:", pairwise.mean().item())

            graph = Data(x=nodes_tensor, edge_index=edge_index).to(device)

            # G = nx.DiGraph()
            # edge_index = graph.edge_index.cpu().numpy()
            # for i in range(edge_index.shape[1]):
            #     G.add_edge(edge_index[0, i], edge_index[1, i])
            # plt.figure(figsize=(8,6))
            # nx.draw(G, node_size=50, arrows=True)
            # plt.title("Graph Structure")
            # plt.show()


            # print("Graph nodes:", nodes_tensor.shape[0])
            # print("Graph edges:", edge_index.shape[1])

            gnn_pred = gnn_model(graph)
            # use video_label not l
            gnn_loss = gnn_lossfn(gnn_pred, video_label.unsqueeze(0))

            print("gnn_loss value:", gnn_loss.item())
            # print("gnn_loss requires_grad:", gnn_loss.requires_grad)
            # print("gnn_loss grad_fn:", gnn_loss.grad_fn)

            # print("GNN pred logits:", gnn_pred)
            # print("GNN pred class:", gnn_pred.argmax(1).item())
            # print("True label:", video_label.item())

            num_chunks = (F_len + chunk_size - 1) // chunk_size
            total_loss = (allcnnlosses / num_chunks) + 0.5 * gnn_loss

            #total_loss = allcnnlosses + 0.5 * gnn_loss
            print("allcnnlosses", allcnnlosses)
            print("total_loss", total_loss)
            total_loss.backward()

            # for name, param in gnn_model.named_parameters():
            #     if param.grad is not None:
            #         print(name, "grad mean:", param.grad.abs().mean().item())
            #     else:
            #         print(name, "NO GRAD")

            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)

            gnn_optimiser.step()
            cnn_optimiser.step()

            # for p in gnn_model.parameters():
            #     if p.grad is not None:
            #         print(p.grad.norm())

            gnn_losses_for_plot.append(gnn_loss.item())
            cnn_losses_for_plot.append(allcnnlosses.item())
            total_losses_for_plot.append(total_loss.item())

            del nodes_tensor, graph, outputs, gnn_pred
            torch.cuda.empty_cache()
    gnn_scheduler.step()
    # if epoch % 5 == 0:
    #     evaluate(cnn_model, graphs_test_loader, epoch)
    #     cnn_model.train()

print("per epoch, the classes that spiked the train loss the most", loss_spiking_classes_per_epoch)

_, ax = plt.subplots(1,1)
ax.set_xlabel('Iter')
ax.set_ylabel('Total loss')
ax.set_title("Total loss training")
# Raw loss (optional, can make it lighter)
ax.plot(total_losses_for_plot, alpha=0.3, label="Raw Loss")
# Smoothed loss
smoothed = moving_average(total_losses_for_plot, window=20)
ax.plot(range(len(smoothed)), smoothed, label="Smoothed Loss", linewidth=2)
ax.legend()
plt.savefig("Total.png")
plt.show()

_, ax = plt.subplots(1,1)
ax.set_xlabel('Iter')
ax.set_ylabel('GNN loss')
ax.set_title(f"GNN loss training")
# Raw loss (optional, can make it lighter)
ax.plot(gnn_losses_for_plot, alpha=0.3, label="Raw gnn Loss") #
# Smoothed loss
smoothed_gnn = moving_average(gnn_losses_for_plot, window=20)
ax.plot(range(len(smoothed_gnn)), smoothed_gnn, label="Smoothed GNN Loss", linewidth=2)
ax.legend()
plt.savefig(f"GNN.png")
plt.show()

_, ax = plt.subplots(1,1)
ax.set_xlabel('Iter')
ax.set_ylabel('CNN loss')
ax.set_title(f"CNN loss training")
# Raw loss (optional, can make it lighter)
ax.plot(cnn_losses_for_plot, alpha=0.3, label="Raw cnn Loss")
# Smoothed loss
smoothed_cnn = moving_average(cnn_losses_for_plot, window=20)
ax.plot(range(len(smoothed_cnn)), smoothed_cnn, label="Smoothed CNN Loss", linewidth=2)
ax.legend()
plt.savefig(f"CNN.png")
plt.show()


training_end_time = time.time()
train_time = training_end_time - training_start_time
print(f"Train time is {train_time}")


test_start_time = time.time()


## Testing using GNN classifier
print("Testing using GNN classifier")
gnn_model.eval()
cnn_model.eval()
test_loss = 0
correct_frames = 0  # frame-level accuracy
video_probs_dict = {}  # accumulate frame probs per video
all_preds = []
all_targets = []
allframes = 0
correct_videos = 0

testing_gnn_losses_plot = []

with torch.no_grad():
    for f, m, l in graphs_test_loader:
        f = f.float() / 255.0          # convert uint8 -> float32 and normalize
        f = f.permute(0, 1, 4, 2, 3)  # (B, F, H, W, C) -> (B, F, C, H, W)

        for vid_idx in range(f.size(0)):
            video_frames = f[vid_idx]
            video_label = l[vid_idx].to(device)
            video_masks = m[vid_idx]

            chunk_size = 30
            F_len = video_frames.shape[0]

            # Initialise once per vid
            subimages_per_frame = {}
            global_frame_count = 0
            allcnnlosses = 0.0
            # make sure init edges once per video, outside all inner loops
            edges_connecting_from = []
            edges_connecting_to = []

            for i in range(0, F_len, chunk_size):
                chunk = video_frames[i:i+chunk_size].to(device)
                chunk = torchfunc.interpolate(chunk, size=(224, 224),
                                            mode='bilinear', align_corners=False)

                outputs, feature_maps = cnn_model(chunk)

                video_labels_chunk = video_label.repeat(chunk.size(0))
                loss = cnn_lossfn(outputs, video_labels_chunk)
                allcnnlosses = allcnnlosses + loss

                for j in range(chunk.size(0)):
                    frame_idx = i + j
                    subimages_per_frame[global_frame_count] = []

                    mask = video_masks[frame_idx].float()
                    mask_np = mask.cpu().numpy().astype("uint8")
                    labeled = label(mask_np, connectivity=2)
                    regions = regionprops(labeled)

                    # Mask at (28,28) to match backbone feature spatial size
                    mask_resized = torchfunc.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(28, 28), mode='nearest'
                    ).squeeze()  # (28, 28)

                    backbone_features = feature_maps[j].unsqueeze(0)  # (1, 512, 28, 28)

                    orig_h, orig_w = video_frames[frame_idx].shape[1:]
                    scale_x = 28 / orig_w
                    scale_y = 28 / orig_h

                    if len(regions) == 0:
                        # fallback node so frame is never empty,, not sure
                        masked_feat = backbone_features.squeeze(0) * mask_resized.to(device)  # (512, 28, 28)
                        masked_feat = torchfunc.adaptive_avg_pool2d(masked_feat, (7, 7))       # (512, 7, 7)
                        subimages_per_frame[global_frame_count].append(masked_feat)
                    else:
                        for rs in regions:
                        # MAX_REGIONS = 5
                        # for rs in regions[:MAX_REGIONS]: 
                            min_row, min_col, max_row, max_col = rs.bbox
                            if (max_row - min_row) < 1 or (max_col - min_col) < 1:
                                continue

                            bbox = torch.tensor([[
                                0,
                                min_col * scale_x,
                                min_row * scale_y,
                                max_col * scale_x,
                                max_row * scale_y
                            ]], dtype=torch.float32).to(device)

                            feat_map = backbone_features.squeeze(0).detach().cpu()  # [1024,14,14]
                            mean_map = feat_map.mean(dim=0)
                            masked_feat = feat_map * mask_resized.cpu()  # (1024,28,28)
                            # # Sum or mean only over masked region
                            mean_map = masked_feat.mean(dim=0)

                            # output_size=(7,7) or (28,28)
                            pooled = roi_align(
                                input=backbone_features,
                                boxes=bbox,
                                output_size=(7, 7)
                            ).squeeze(0)  # (512, 7, 7)
                            subimages_per_frame[global_frame_count].append(pooled)

                        # degenerate bbox fallback-- double check
                        if len(subimages_per_frame[global_frame_count]) == 0:
                            masked_feat = backbone_features.squeeze(0) * mask_resized.to(device)
                            masked_feat = torchfunc.adaptive_avg_pool2d(masked_feat, (7, 7))
                            subimages_per_frame[global_frame_count].append(masked_feat)

                    global_frame_count += 1
                # end of frame loop
            # end of chunk loop

            # graph after all chunks/frames processed
            frame_ids = sorted(subimages_per_frame.keys())
            frame_offsets = {}
            offset = 0
            for fid in frame_ids:
                frame_offsets[fid] = offset
                offset += len(subimages_per_frame[fid])

            all_nodes = []
            for idx, fid in enumerate(frame_ids):
                nodes_this_frame = subimages_per_frame[fid]
                all_nodes.extend(nodes_this_frame)

                if idx == len(frame_ids) - 1:
                    continue

                next_fid = frame_ids[idx + 1]
                for ci in range(len(nodes_this_frame)):
                    src = frame_offsets[fid] + ci
                    for ni in range(len(subimages_per_frame[next_fid])):
                        tgt = frame_offsets[next_fid] + ni
                        edges_connecting_from.append(src)
                        edges_connecting_to.append(tgt)

            if len(all_nodes) == 0:
                print(f"Skipping video {vid_idx} — no nodes")
                continue

            nodes_tensor = torch.stack(all_nodes)  # (N, 512, 7, 7) — all same shape
            edge_index = torch.tensor(
                [edges_connecting_from, edges_connecting_to], dtype=torch.long
            ).to(device)

            x_flat = nodes_tensor.mean(dim=(2,3))  # (N, 512)
            pairwise = torch.cdist(x_flat, x_flat)

            graph = Data(x=nodes_tensor, edge_index=edge_index).to(device)
            gnn_pred = gnn_model(graph)

            video_labels = video_label.repeat(video_frames.size(0))
            test_gnn_loss = gnn_lossfn(gnn_pred, video_label.unsqueeze(0)).item()
            
            test_loss += test_gnn_loss
            testing_gnn_losses_plot.append(test_gnn_loss)
    
            # predicted class
            _, predicted = torch.max(gnn_pred, 1)

            # for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(video_label.unsqueeze(0).cpu().numpy())
            
            # vid-level accuracy for GNN
            correct_videos += (gnn_pred.argmax(1) == video_label.unsqueeze(0)).type(torch.float).sum().item()
            
            
size = len(graphs_test_loader.dataset) # num of videos
print("size is", size)
video_acc = correct_videos / size
test_loss /= len(graphs_test_loader)

print(f"Video Accuracy: {100*video_acc:.1f}%, Loss: {test_loss:.6f}")
test_end_time = time.time()
test_time = test_end_time - test_start_time
print(f"Test time is {test_time}")

cm = confusion_matrix(all_targets, all_preds)
# print("Confusion Matrix:")
# print(cm)

per_class_acc = cm.diagonal() / cm.sum(axis=1)

print("\nPer-class accuracy:")
for i, acc in enumerate(per_class_acc):
    print(f"Class {i}: {acc*100:.2f}%")

_, ax = plt.subplots(1,1)
ax.set_xlabel('Iter')
ax.set_ylabel('Test GNN loss')
ax.set_title(f"Test GNN loss training")
# Raw loss (optional, can make it lighter)
ax.plot(testing_gnn_losses_plot, alpha=0.3, label="Raw gnn Loss")
# Smoothed loss
smoothed_gnn_test = moving_average(testing_gnn_losses_plot, window=20)
ax.plot(range(len(smoothed_gnn_test)), smoothed_gnn_test, label="Smoothed GNN Loss", linewidth=2)
ax.legend()
plt.savefig(f"Test_GNN.png")
plt.show()


# ------ Testing using CNN classifier
# print("Testing using CNN classifier")
# cnn_model.eval()
# test_loss = 0
# correct_frames = 0  # frame-level accuracy
# video_probs_dict = {}  # accumulate frame probs per video
# all_preds = []
# all_targets = []
# allframes = 0


# testing_cnn_losses_plot = []

# with torch.no_grad():
#     for f, r, l in graphs_test_loader:
#         f = f.float() / 255.0          # convert uint8 -> float32 and normalize
#         f = f.permute(0, 1, 4, 2, 3)  # (B, F, H, W, C) -> (B, F, C, H, W)
        
#         for vid_idx in range(f.size(0)):
#             allframes += f.size(1)
#             video_frames = f[vid_idx]           # (F, H, W, C)
#             video_label = l[vid_idx]            # scalar class for this video
#             videoregions = r[vid_idx]

#             frames = video_frames.to(device)
#             video_label = video_label.to(device)

#             video_labels = video_label.repeat(video_frames.size(0))

#             pred, _ = cnn_model(frames)  # (batch_size, num_classes)
#             # print("pred argmax", pred.argmax(1))
#             # print("target", target)
#             test_cnn_loss = cnn_lossfn(pred, video_labels).item()
#             test_loss += test_cnn_loss
#             testing_cnn_losses_plot.append(test_cnn_loss)

#             # predicted class
#             _, predicted = torch.max(pred, 1)

#             # for confusion matrix
#             all_preds.extend(predicted.cpu().numpy())
#             all_targets.extend(video_labels.cpu().numpy())
            
#             # frame-level accuracy
#             # print("gnn pred argmax", pred.argmax(1))
#             # print("video labels", video_labels)
#             correct_frames += (pred.argmax(1) == video_labels).type(torch.float).sum().item()
            
#             # softmax probabilities
#             probs = torchfunc.softmax(pred, dim=1)
            
#             # accumulate per video (using target as video ID)
#             for i, vid in enumerate(video_labels):
#                 vid = vid.item()
#                 if vid not in video_probs_dict:
#                     video_probs_dict[vid] = []
#                 video_probs_dict[vid].append(probs[i].cpu())

# # compute video-level predictions
# correct_videos = 0
# for vid, prob_list in video_probs_dict.items():
#     avg_prob = torch.stack(prob_list).mean(dim=0)
#     pred_class = avg_prob.argmax().item()
#     if pred_class == vid:  # video-level prediction matches target
#         correct_videos += 1

# size = len(graphs_test_loader.dataset) # num of videos
# num_videos = len(video_probs_dict)
# print("correct frames is", correct_frames)
# print("size is", size)
# print("all frames is", allframes)
# # frame_acc = correct_frames / size
# frame_acc = correct_frames / allframes
# video_acc = correct_videos / num_videos
# test_loss /= len(graphs_test_loader)

# print(f"Frame Accuracy: {100*frame_acc:.1f}%, Video Accuracy: {100*video_acc:.1f}%, Loss: {test_loss:.6f}")
# test_end_time = time.time()
# test_time = test_end_time - test_start_time
# print(f"Test time is {test_time}")

# cm = confusion_matrix(all_targets, all_preds)
# # print("Confusion Matrix:")
# # print(cm)

# per_class_acc = cm.diagonal() / cm.sum(axis=1)

# print("\nPer-class accuracy:")
# for i, acc in enumerate(per_class_acc):
#     print(f"Class {i}: {acc*100:.2f}%")

# _, ax = plt.subplots(1,1)
# ax.set_xlabel('Iter')
# ax.set_ylabel('Test CNN loss')
# ax.set_title(f"Test CNN loss training")
# # Raw loss (optional, can make it lighter)
# ax.plot(testing_cnn_losses_plot, alpha=0.3, label="Raw cnn Loss")
# # Smoothed loss
# smoothed_cnn_test = moving_average(testing_cnn_losses_plot, window=20)
# ax.plot(range(len(smoothed_cnn_test)), smoothed_cnn_test, label="Smoothed CNN Loss", linewidth=2)
# ax.legend()
# plt.savefig(f"Test_CNN.png")
# plt.show()
evaluate(cnn_model, graphs_test_loader)
