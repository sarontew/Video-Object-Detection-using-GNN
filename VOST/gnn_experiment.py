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
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from cnn_baseline import FrameClassifier
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from skimage.measure import label, regionprops
from gnn_dataset import New_GNN_Dataset
from torchvision.models.detection.roi_heads import roi_align

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
class St_GCN_Classifier(torch.nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        self.hidden_channels = 64
        self.conv1 = GCNConv(512,self.hidden_channels)
        #self.conv1 = GCNConv(2048,self.hidden_channels) # input feature per node = 2, output feature per node =4
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.classifier = nn.Linear(self.hidden_channels, num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = x.mean(dim=(2,3))  # collapse ROI feature maps
        #x = x.flatten(2).mean(dim=2) alt

        x = self.conv1(x, edge_index)
        x = torchfunc.relu(x)

        x = self.conv2(x, edge_index)
        x = torchfunc.relu(x)

        x = global_mean_pool(x, batch)

        x = self.classifier(x)
        return x

training_start_time = time.time()

all_valid_file_names = get_file_names('train.txt') + get_file_names('val.txt')
#all_valid_file_names = all_valid_file_names[0:2]

#all_valid_file_names = ['1186_cut_chilli', '4176_cut_cloth', '4174_cut_cloth', '4331_cut_cloth', '4320_tear_dough', '226_squeeze_dough', '2218_empty_raisin', '455_fold_box',  '1184_cut_chilli']
all_valid_file_names = ['455_fold_box', '4320_tear_dough', '226_squeeze_dough', '3705_flatten_box']

class_to_files = defaultdict(list)

for file in all_valid_file_names:
    olabel = get_object_label(file)
    class_to_files[olabel].append(file)

all_classes = list(class_to_files.keys())
#print("all classes", all_classes)
random.shuffle(all_classes)
overlap_fraction = 1 # was 0.18
num_overlap_classes = max(1, int(len(all_classes) * overlap_fraction))
overlap_classes = set(all_classes[:num_overlap_classes])
#print("overlap classes are", overlap_classes)
train_files = []
test_files = []
for cls, files in class_to_files.items():
    random.shuffle(files)
    if cls in overlap_classes:
        split_idx = int(0.8 * len(files))
        train_files.extend(files[:split_idx])
        test_files.extend(files[split_idx:])
    else:
        # Assign whole class to train (or randomly choose)
        train_files.extend(files)

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
gnn_optimiser = torch.optim.Adam(gnn_model.parameters(), lr=0.001) ## too large
cnn_optimiser = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
cnn_lossfn = torch.nn.CrossEntropyLoss()
gnn_lossfn = torch.nn.CrossEntropyLoss()

seen_labels = []
total_losses_for_plot = []
gnn_losses_for_plot = []
cnn_losses_for_plot = []


for epoch in range(1):
    print("epoch", epoch)
    gnn_model.train()
    cnn_model.train()

    total_loss = 0
    total_gnn_loss = 0
    num_vids = 0
    

    for f, m, l in graphs_train_loader: # for each video
        print("graph label", l)
        print("COUNTT", num_vids)
        all_graphs = []
        num_vids+= 1

        #seen_labels.append(l.item())

        f = f.float() / 255.0          # convert uint8 -> float32 and normalize
        f = f.permute(0, 1, 4, 2, 3)  # (B, F, H, W, C) -> (B, F, C, H, W)
        
        #trying to total losses 
        gnn_optimiser.zero_grad()
        cnn_optimiser.zero_grad()
        allcnnlosses = 0.0
        total_gnn_losses = 0.0
        all_cnn_losses_list = []

        #print("gonna loop through range", f.size(0))

        for vid_idx in range(f.size(0)): # for one video
            #print("loop 2, vid idx", vid_idx)
            video_frames = f[vid_idx]           # (F, H, W, C)
            video_label = l[vid_idx]            # scalar class for this video
            video_masks = m[vid_idx]

            chunk_size = 30
            F = video_frames.size(1)

            for i in range(0, F, chunk_size):
                chunk = video_frames[i:i+chunk_size]  # shape: (<=10, H, W, C)
                chunk = torchfunc.interpolate(
                        chunk,
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )
                #print(f"Chunk {i//chunk_size} shape: {chunk.shape}")

                chunk = chunk.to(device)
                video_label = video_label.to(device)

                outputs, feature_maps = cnn_model(chunk)
                #print("cnn gives", feature_maps.shape)
                
                # label is per video, repeat it for each frame
                video_labels = video_label.repeat(chunk.size(0))

                loss = cnn_lossfn(outputs, video_labels)
                allcnnlosses = allcnnlosses + loss


                frame_count = -1
                subimages_per_frame = {}

                for j in range(chunk.size(0)):
                    frame_idx = i + j
                    
                    frame_count+= 1
                    subimages_per_frame[frame_count] = []

                    frame = video_frames[frame_idx]
                    #print("frame dim", frame.shape) # 3, 1080, 1440 or 3,1080,1920 or

                    frame_np = frame.permute(1, 2, 0).cpu().numpy()
                    
                    mask = video_masks[frame_idx]
                    #print("ININITT mask shape", mask.shape)
                    mask_np = mask.cpu().numpy().astype("uint8")
                    roi = cv2.bitwise_and(frame_np, frame_np, mask=mask_np)

                    # cv2.imshow("roi", roi)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    labeled = label(mask_np, connectivity=2) # 2 = 8 connectivity, type:ndarray
                    regions = regionprops(labeled) # type: list

                    mask = mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,224,224)
                    #print("mask init", mask.shape)

                    mask_resized = torchfunc.interpolate(
                        mask,
                        size=(28, 28),
                        mode='nearest'  # important for masks
                    )  # (1,1,14,14)

                    mask_resized = mask_resized.squeeze()  # (14,14)

                    # print("mask resized", mask_resized.shape) # 1024,14,14
                    # print("feature_maps[j]", feature_maps[j].shape) # 1024,14,14

                    masked_features = feature_maps[j].to(device) * mask_resized.to(device)  # broadcast to (1024,14,14)
                    region_feature = masked_features.mean(dim=(1,2))  # (1024,)
                    #print("region features", region_feature.shape)
                    #print("region_feat mean:", region_feature.mean().item())
                    #subimages_per_frame[frame_count].append(region_feature) # 1024
                    
                    for rs in regions:
                        min_row, min_col, max_row, max_col = rs.bbox
                        subimage = roi[min_row:max_row, min_col:max_col]

                        frame_tensor = torch.from_numpy(frame_np).permute(2,0,1).unsqueeze(0).float() #([1, 3, 224, 224])
                        #print("frame_tensor shape", frame_tensor.shape)

                        # not sure backbone_features = feature_maps[frame_idx].unsqueeze(0) #get frames from cnn backbone feats
                        backbone_features = feature_maps[j].unsqueeze(0) # torch.Size([1, 1024, 14, 14])
                        #print("backbone_features shape", backbone_features.shape)


                        # PLOTs
                        # feature_grid = np.zeros((224, 224))

                        # x1 = int(min_col)
                        # y1 = int(min_row)
                        # x2 = int(max_col)
                        # y2 = int(max_row)

                        # feature_grid[y1:y2, x1:x2] = 1

                        # plt.imshow(feature_grid, cmap='jet')
                        # plt.title("Before scaling BBox on Feature Map (14x14)")
                        # plt.colorbar()
                        # plt.show()


                        orig_h, orig_w = frame_np.shape[:2]

                        scale_x = 28 / orig_w ## backbone feat in 14,14 dim for the 224,224 frame image
                        scale_y = 28 / orig_h

                        bbox = torch.tensor([[
                            0,
                            min_col * scale_x,
                            min_row * scale_y,
                            max_col * scale_x,
                            max_row * scale_y
                        ]], dtype=torch.float32).to(device)

                        ### Area
                        width = bbox[0, 3] - bbox[0, 1]
                        height = bbox[0, 4] - bbox[0, 2]

                        area = width * height
                        print("box area is", area)

                        feature_grid = np.zeros((28, 28))

                        x1 = int(min_col * scale_x)
                        y1 = int(min_row * scale_y)
                        x2 = int(max_col * scale_x)
                        y2 = int(max_row * scale_y)

                        feature_grid[y1:y2, x1:x2] = 1

                        # plt.imshow(feature_grid, cmap='jet')
                        # plt.title("BBox on Feature Map (14x14)")
                        # plt.colorbar()
                        # plt.show()

                        feat_map = backbone_features.squeeze(0).detach().cpu()  # [1024,14,14]

                        #mean_map = feat_map.mean(dim=0)

                        masked_feat = feat_map * mask_resized.cpu()  # (1024,28,28)

                        # Sum or mean only over masked region
                        mean_map = masked_feat.mean(dim=0)

                        # plt.imshow(mean_map, cmap='viridis')
                        # plt.title("Mean Feature Activation")
                        # plt.colorbar()
                        # plt.show()

                        heatmap = mean_map.unsqueeze(0).unsqueeze(0)  # [1,1,14,14]
                        heatmap = torchfunc.interpolate(heatmap, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                        heatmap = heatmap.squeeze().numpy()

                        # plt.imshow(frame_np)
                        # plt.imshow(heatmap, cmap='jet', alpha=0.5)
                        # plt.title("Feature Attention Overlay")
                        # plt.colorbar()
                        # plt.show()

                        #print("final bbox", bbox)

                        # print("bbox:", bbox)
                        # print("feature map size:", backbone_features.shape) # 1,2048,7,7
                        # if (max_row - min_row) == 0 or (max_col - min_col) == 0:
                        #     print("Invalid bbox:", min_row, min_col, max_row, max_col)
                        # print("mask sum:", mask_np.sum())
                        # print("feature map mean:", feature_maps.mean().item())
                        # print("feature map max:", feature_maps.max().item())
                        # print("before ROI mean:", backbone_features.mean().item())
                        
                        pooled = roi_align(input=backbone_features, boxes=bbox, output_size=(28,28)) # 1, 2048, 7, 7
                        pooled = pooled.squeeze(0) # remove the first dim
                        #print("after ROI mean:", pooled.mean().item())
                        #print("pooled dim", pooled.shape) # 1024,7,7 cuz of output shape

                        subimages_per_frame[frame_count].append(pooled)
                        roi_map = pooled.mean(dim=0).detach().cpu()

                        # normalize
                        roi_map -= roi_map.min()
                        roi_map /= (roi_map.max() + 1e-8)

                        # plt.imshow(roi_map, cmap='jet')
                        # plt.title("ROI Align Feature Map (28x28)")
                        # plt.colorbar()
                        # plt.show()

                        # Step 1: upsample ROI feature map to bbox size
                        # roi_up = torchfunc.interpolate(
                        #     roi_map.unsqueeze(0).unsqueeze(0),
                        #     size=(max_row - min_row, max_col - min_col),
                        #     mode='bilinear',
                        #     align_corners=False
                        # ).squeeze().numpy()

                        # # Step 2: normalize
                        # roi_up -= roi_up.min()
                        # roi_up /= (roi_up.max() + 1e-8)

                        # # Step 3: create empty full-image heatmap
                        # full_heatmap = np.zeros((orig_h, orig_w))

                        # # Step 4: place ROI heatmap into correct location
                        # full_heatmap[min_row:max_row, min_col:max_col] = roi_up

                        # # Step 5: overlay on full image
                        # plt.imshow(frame_np.astype(np.uint8))
                        # plt.imshow(full_heatmap, cmap='jet', alpha=0.5)

                        # # draw bbox for reference
                        # plt.gca().add_patch(
                        #     plt.Rectangle(
                        #         (min_col, min_row),
                        #         max_col - min_col,
                        #         max_row - min_row,
                        #         edgecolor='white',
                        #         linewidth=2,
                        #         fill=False
                        #     )
                        # )

                        # plt.title("ROI Align Feature Overlay (Full Image)")
                        # plt.axis('off')
                        # plt.show()

                    edges_connecting_from = []
                    edges_connecting_to = []
                    sub_images_as_nodes = []

                    # Fix frame offset for the edge node count
                    frame_ids = sorted(subimages_per_frame.keys())
                    frame_offsets = {}
                    offset = 0
                    for frame_id in frame_ids:
                        frame_offsets[frame_id] = offset
                        offset += len(subimages_per_frame[frame_id])

                    # Create nodes and edges
                    for idx, frame_id in enumerate(frame_ids):
                        sub_images = subimages_per_frame[frame_id]
                        sub_images_as_nodes.extend(sub_images) # flat nodes

                        if idx == len(frame_ids) - 1: # skip last frame
                            continue

                        next_frame_id = frame_ids[idx + 1]
                        next_frame_subimages = subimages_per_frame[next_frame_id]

                        for current_subimage_index in range(len(sub_images)):
                            source_node = frame_offsets[frame_id] + current_subimage_index

                            for next_frame_index in range(len(next_frame_subimages)): # create an edge for all of next frame's subimages
                                target_node = frame_offsets[next_frame_id] + next_frame_index

                                edges_connecting_from.append(source_node)
                                edges_connecting_to.append(target_node)

        all_nodes = []
        for frame_id in subimages_per_frame:
            for pooled in subimages_per_frame[frame_id]:
                #print("creating nodes, pooled dim is", pooled.shape)
                all_nodes.append(pooled)
        nodes = torch.stack(all_nodes)
        print("This graph has nodes", len(all_nodes))
        #print("graph has been created for this video")

        graph = Data(x=nodes, edge_index=torch.tensor([edges_connecting_from, edges_connecting_to], dtype=torch.long))

        
        ### graph debug plots
        G = nx.DiGraph()

        edge_index = graph.edge_index.cpu().numpy()

        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])

        plt.figure(figsize=(8,6))
        nx.draw(G, node_size=50, arrows=True)
        plt.title("Graph Structure")
        plt.show()


        plt.imshow(frame_np.astype(np.uint8))

        node_colors = []

        for frame_id in frame_ids:
            num_nodes = len(subimages_per_frame[frame_id])
            node_colors.extend([frame_id] * num_nodes)

        plt.figure(figsize=(8,6))
        nx.draw(G, node_color=node_colors, cmap='viridis', node_size=50)
        plt.title("Graph Colored by Frame")
        plt.show()

        node_positions = []

        for frame_id in frame_ids:
            for rs in regionprops(label(video_masks[frame_id].cpu().numpy())):
                min_row, min_col, max_row, max_col = rs.bbox
                
                cx = (min_col + max_col) / 2
                cy = (min_row + max_row) / 2
                
                node_positions.append((cx, cy))
                plt.scatter(cx, cy, c='red', s=20)

        plt.title("Graph Nodes on Image")
        plt.show()

        gloader = GeoDataLoader([graph], batch_size=1, shuffle=True)

        graph = graph.to(device)
        l = l.to(device)
        #print("training graph w label", l)
        
        #x, edge_index, batch
        param = next(cnn_model.parameters())
        before = param.clone().detach()

        gnn_pred = gnn_model(graph)

        gnn_loss = gnn_lossfn(gnn_pred, l)

        total_loss = allcnnlosses + gnn_loss

        gnn_losses_for_plot.append(gnn_loss.item())
        cnn_losses_for_plot.append(allcnnlosses.item())

        print("cnn loss is", allcnnlosses) 
        print("gnn loss is", gnn_loss) 
        print("total loss is", total_loss)
        
        total_loss.backward()
        #print("graph grad = 0",(graph.x.grad != 0).any())
        # print("graph x mean", graph.x.mean())
        # print("gnn pred", gnn_pred)
        # print("edges info", graph.edge_index.max(), graph.x.shape[0])
        # print("graph x mean:", graph.x.grad.abs().mean().item())
        # print("graph x max:", graph.x.grad.abs().max().item())

        for name, param in cnn_model.named_parameters():
            if param.grad is not None:
                print(name, "grad mean:", param.grad.abs().mean().item(),
                            "grad max:", param.grad.abs().max().item())
                break  # just check first layer (or remove break to see all)


        gnn_optimiser.step()
        cnn_optimiser.step()

        total_losses_for_plot.append(total_loss.item())

        after = param.clone().detach()
        print("CNN from GNN Weights changed:", not torch.equal(before, after))
        diff = (before - after).abs().mean().item()
        print("CNN from GNN Mean weight change:", diff)
        #print("seen labels", seen_labels)

        total_gnn_loss += gnn_loss.item()

        print(torch.cuda.memory_allocated() / 1e9, "GB")
        del feature_maps, nodes, graph, outputs, loss, gnn_pred
        torch.cuda.empty_cache()


## plot loss
_, ax = plt.subplots(1,1)
ax.set_xlabel('Iter')
ax.set_ylabel('Total loss')
ax.set_title(f"Total loss training")
plt.plot(range(len(total_losses_for_plot)), total_losses_for_plot, label="Total Train loss")
plt.savefig(f"Total.png")
plt.show()

_, ax = plt.subplots(1,1)
ax.set_xlabel('Iter')
ax.set_ylabel('GNN loss')
ax.set_title(f"GNN loss training")
plt.plot(range(len(gnn_losses_for_plot)), gnn_losses_for_plot, label="GNN Train loss")
plt.savefig(f"GNN.png")
plt.show()

_, ax = plt.subplots(1,1)
ax.set_xlabel('Iter')
ax.set_ylabel('CNN loss')
ax.set_title(f"CNN loss training")
plt.plot(range(len(cnn_losses_for_plot)), cnn_losses_for_plot, label="CNN Train loss")
plt.savefig(f"CNN.png")
plt.show()


training_end_time = time.time()
train_time = training_end_time - training_start_time
print(f"Train time is {train_time}")


test_start_time = time.time()
## testtt
cnn_model.eval()
test_loss = 0
correct_frames = 0  # frame-level accuracy
video_probs_dict = {}  # accumulate frame probs per video
all_preds = []
all_targets = []
allframes = 0


with torch.no_grad():
    for f, r, l in graphs_test_loader:
        f = f.float() / 255.0          # convert uint8 -> float32 and normalize
        f = f.permute(0, 1, 4, 2, 3)  # (B, F, H, W, C) -> (B, F, C, H, W)
        
        for vid_idx in range(f.size(0)):
            allframes += f.size(1)
            video_frames = f[vid_idx]           # (F, H, W, C)
            video_label = l[vid_idx]            # scalar class for this video
            videoregions = r[vid_idx]

            frames = video_frames.to(device)
            video_label = video_label.to(device)

            video_labels = video_label.repeat(video_frames.size(0))

            pred, _ = cnn_model(frames)  # (batch_size, num_classes)
            # print("pred argmax", pred.argmax(1))
            # print("target", target)
            test_loss += cnn_lossfn(pred, video_labels).item()

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
