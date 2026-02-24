import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
import torch_geometric
from torch import Tensor
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import save_region_features
from torchvision import transforms, models
from torchvision.models import resnet50
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from cnn_baseline import CNN_Classifier
from torch_geometric.loader import DataLoader
from skimage.measure import label, regionprops
from torchvision.models.detection.roi_heads import roi_align

# video = "0_squeeze_cloth"
video = "1214_cut_mellon"
frame_count = 0
subimages_per_frame = {} # frame_number : [subimages]

resnet = models.resnet50(pretrained=True)
backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
backbone.eval()

# @profile
for image_name in os.listdir(f"JPEGImages/{video}"):
    subimages_per_frame[frame_count] = []
    
    if frame_count != -1:

        frame = cv2.imread(f"JPEGImages/{video}/{image_name}")
        print("frame")
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        mask_name = image_name[0:-3] + "png"

        mask_parent_dir = "Annotations"
        mask = cv2.imread(f"{mask_parent_dir}/{video}/{mask_name}", cv2.IMREAD_GRAYSCALE)
        mask_gray = mask[:, :, 0] if len(mask.shape) == 3 else mask

        # Robust threshold: anything > 0 is ROI
        binary_mask = np.where(mask_gray > 0, 255, 0).astype(np.uint8)
        roi = cv2.bitwise_and(frame, frame, mask=binary_mask)

        labeled = label(binary_mask, connectivity=2) # 2 = 8 connectivity, type:ndarray
        regions = regionprops(labeled) # type: list

        # Get each region's sub-image from masked out
        for r in regions:
            min_row, min_col, max_row, max_col = r.bbox
            subimage = roi[min_row:max_row, min_col:max_col]
            # cv2.imshow("subimage", subimage)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print("subimage type is", type(subimage))
            # print("with shape", subimage.shape)

            #resnet = models.resnet50(pretrained=True)
            # Remove avgpool and fc
            #backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
            #x = torch.randn(1, 3, 224, 224)
            #feature_map = backbone(x)

            frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float()
            with torch.no_grad():
                backbone_features = backbone(frame_tensor)
                bbox = torch.tensor([[0, min_col, min_row, max_col, max_row]], dtype=torch.float32)
                
                pooled = roi_align(input=backbone_features, boxes=bbox, output_size=(7,7)) # 1, 2048, 7, 7
                pooled = pooled.squeeze(0) # remove the first dim

                #subimages_per_frame[frame_count].append(subimage) # each chunk from each frame
                # add the pooled features instead
                subimages_per_frame[frame_count].append(pooled.cpu())

        
        # plt.imshow(labeled, cmap="tab20")
        # plt.colorbar()
        # plt.show()

        frame_count += 1
        print(frame_count)

print("done extracting feature for each subimage of interest")

# Graph creation
# Node: each sub-images from every single frame
# Edge: connecting subimages from frame i to every subimage from frame i+1

# print("before creating edges, checking the node features i have saved")
# for subimage in subimages_per_frame.items():
#     print(len(subimage))
#     for s in subimage:
#         print("s is ",type(s))


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

# Hs = [img.shape[0] for img in sub_images_as_nodes]
# Ws = [img.shape[1] for img in sub_images_as_nodes]
# C  = sub_images_as_nodes[0].shape[2]

# max_H = max(Hs)
# max_W = max(Ws)

# padded_images = []

# for img in sub_images_as_nodes:
#     assert img.ndim == 3, img.shape  # (H, W, C)
#     H, W, C = img.shape
#     print("shape")
#     print(H,W,C)

#     cv2.imshow("subimage featurez", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     pad_h = max_H - H
#     pad_w = max_W - W

#     # pad format: ((top,bottom), (left,right), (channels))
#     padded = np.pad(
#         img,
#         pad_width=((0, pad_h), (0, pad_w), (0, 0)),
#         mode="constant",  # or "edge", "reflect", etc.
#         constant_values=0
#     )
#     # print("padded shape", padded.shape)
#     # cv2.imshow("subimage", padded)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     assert padded.shape == (max_H, max_W, C)
#     padded_images.append(padded)

#batch = torch.from_numpy(np.stack(padded_images))

#FEATURE BASED BATCH
# print("now after creating edges, before creating batch")
# for k, subimage in subimages_per_frame.items():
#     print(len(subimage))
#     for s in subimage:
#         print(type(s), k)
        # if type(s) == list:
        #     print("getting list type for key", k)
        # elif type(s) == int:
        #     print("getting CORRECT int type for key", k)

# print(type(subimages_per_frame.values()))
# nodes = torch.from_numpy(np.stack(subimages_per_frame.values()))

# NEW: flatten to make everything even - NOT SURE
all_nodes = []
for frame_id in subimages_per_frame:
    for pooled in subimages_per_frame[frame_id]:
        all_nodes.append(pooled)
nodes = torch.stack(all_nodes) 

video_1_graph = Data(x=nodes, edge_index=torch.tensor([edges_connecting_from, edges_connecting_to], dtype=torch.long))
g = torch_geometric.utils.to_networkx(video_1_graph, to_undirected=True)
plt.figure()
nx.draw(g)
plt.show()

# x: node feature matrix with shape [num_nodes, num_node_features]
# edge_index: graph connectivity in COO format with shape [2, num_edges]
# edge_attr: edge feature matrix with shape [num_edge, num_edge_features]
# y: target to train against e.g. node level targets of shape [num_nodes, *] or graph-level targets of shape [1,*]
# pos: node position matrix with shape [num_nodes, num_dimensions]

video_1_graph.validate(raise_on_error=True)

# print(data.num_nodes)
# print(data.num_edges)
# print(data.has_self_loops())
# print(data.has_isolated_nodes())

class St_GCN_Classifier(torch.nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        # model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # for name, layer in model.named_children():
        #     if name in ['conv1', 'bn1', 'layer1', 'layer2']:
        #         for param in layer.parameters():
        #             param.requires_grad = False
            

        self.hidden_channels = 64
        self.conv1 = GCNConv(2048,self.hidden_channels) # input feature per node = 2, output feature per node =4
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.classifier = nn.Linear(self.hidden_channels, num_classes)

    def forward(self,x, edge_index, batch):

        # Node level message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph level pooling
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


dataset = [video_1_graph] # graph of images not features
cnn_dataset = []
train_loader = DataLoader(dataset, batch_size=64, shuffle = True)

# trying to extract the features of each node
save_region_features(train_loader) # i dont think its actually saving anything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gnn_model = St_GCN_Classifier(num_classes=2).to(device)
cnn_model = CNN_Classifier(num_classes=1).to(device) # TODO: fix num_classes, get unique object/action labels based on videos
gnn_optimiser = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
cnn_optimiser = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

gnn_model.train()
cnn_model.train()
for epoch in range(50):
    total_loss = 0

    for batch in train_loader: # for frame, graph
        batch = batch.to(device)

        gnn_optimiser.zero_grad()
        #cnn_optimiser.zero_grad()

        #cnn_pred = cnn_model() # input the frame
        gnn_pred = gnn_model(batch.x, batch.edge_index, batch.batch)

        gnn_loss = criterion(gnn_pred, batch.y)
        #cnn_loss = criterion(cnn_pred, batch.y)
        
        #all_cnn_loss_per_videos += cnn_loss
        
        gnn_loss.backward()
        gnn_optimiser.step()

        total_gnn_loss += gnn_loss.item()

    print(f"Epoch {epoch:03d}, GNN Loss: {total_gnn_loss:.4f}")


# model = MyGCN()
# out = model(video_1_graph)
# print("Output ndoe features after GN layer is", out)