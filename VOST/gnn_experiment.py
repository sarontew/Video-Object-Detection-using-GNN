import os
import cv2
import torch
import numpy as np
import networkx as nx
import torch_geometric
from torch import Tensor
from scipy import ndimage
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from skimage.measure import label, regionprops

# video = "0_squeeze_cloth"
video = "7_squeeze_pasta"
frame_count = 0
subimages_per_frame = {} # frame_number : [subimages]

for image_name in os.listdir(f"JPEGImages/{video}"):
    frame = cv2.imread(f"JPEGImages/{video}/{image_name}")
    subimages_per_frame[frame_count] = []
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

        subimages_per_frame[frame_count].append(subimage) # each chunk from each frame
    
    # plt.imshow(labeled, cmap="tab20")
    # plt.colorbar()
    # plt.show()

    frame_count += 1

# Graph creation
# Node: each sub-images from every single frame
# Edge: connecting subimages from frame i to every subimage from frame i+1

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

# Not all subimages are the same size - pad
flattened_nodes = [node.reshape(-1) for node in sub_images_as_nodes]
max_len = max(node.shape[0] for node in flattened_nodes)
padded_nodes = np.stack([
    np.pad(node, (0, max_len - node.shape[0]))
    for node in flattened_nodes
])
nodes = torch.tensor(padded_nodes, dtype=torch.float)

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

class MyGCN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = GCNConv(2,4) # input feature per node = 2, output feature per node =4

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.relu(x)
    
# model = MyGCN()
# out = model(video_1_graph)
# print("Output ndoe features after GN layer is", out)