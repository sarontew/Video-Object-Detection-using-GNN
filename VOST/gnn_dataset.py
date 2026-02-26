import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision import transforms, models
from skimage.measure import label, regionprops
from torchvision.models.detection.roi_heads import roi_align


class GNN_Dataset(Dataset):

    def __init__(self, video_names, split='train', task='object_rec'):
        self.all_graphs = []
        self.all_frames = []
        self.all_target_labels = []
        self.unique_label_mappings = {}
        self.init_frame = None
        self.final_frame = None
        self.split = split
        self.video_names = video_names
        self.original_label = []

        for video in video_names:
            graph = self.generate_graph(video)
            self.all_graphs.append(graph)

            if task == 'object_rec':
                target_name = video.split("_")[2] # object
            elif task == 'action_rec':
                target_name = video.split("_")[1] # action
            else:
                raise Exception("Valid task for object or action recognition not provided")
            
            self.original_label.append(target_name)

            if target_name not in self.unique_label_mappings.keys(): # Ensure unique id for each object
                self.unique_label_mappings[target_name] = self.video_names.index(video)
                print("target name is", target_name)
                print("unique label id", self.video_names.index(video))
                print("video id is", video)

    def generate_graph(self, video):
        resnet = models.resnet50(pretrained=True)
        backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
        backbone.eval()
        frame_count = -1
        all_frames = []
        subimages_per_frame = {}

        for name in os.listdir(f"JPEGImages/{video}"):
            all_frames.append(name)

        for image_name in os.listdir(f"JPEGImages/{video}"):
            frame_count += 1

            if self.init_frame==None or self.final_frame==None:
                # First half train, second half test
                if self.split == 'train':
                    self.init_frame = 0
                    self.final_frame = len(all_frames) // 2
                elif self.split == 'test':
                    self.init_frame = len(all_frames) // 2 + 1
                    self.final_frame = len(all_frames) -1

            if self.init_frame!=None and self.final_frame!=None:
                if frame_count <= (self.final_frame - self.init_frame):
                    print("extracting frame", frame_count)

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

                        frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float()
                        with torch.no_grad():
                            backbone_features = backbone(frame_tensor)
                            bbox = torch.tensor([[0, min_col, min_row, max_col, max_row]], dtype=torch.float32)
                            
                            pooled = roi_align(input=backbone_features, boxes=bbox, output_size=(7,7)) # 1, 2048, 7, 7
                            pooled = pooled.squeeze(0) # remove the first dim

                            #subimages_per_frame[frame_count].append(subimage) # each chunk from each frame
                            # add the pooled features instead
                            subimages_per_frame[frame_count].append(pooled.cpu())

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

        all_nodes = []
        for frame_id in subimages_per_frame:
            for pooled in subimages_per_frame[frame_id]:
                all_nodes.append(pooled)
        nodes = torch.stack(all_nodes) 

        print("dataset subimages per frame is", subimages_per_frame.keys())

        video_1_graph = Data(x=nodes, edge_index=torch.tensor([edges_connecting_from, edges_connecting_to], dtype=torch.long))
        # x: node feature matrix with shape [num_nodes, num_node_features]
        # edge_index: graph connectivity in COO format with shape [2, num_edges]
        # edge_attr: edge feature matrix with shape [num_edge, num_edge_features]
        # y: target to train against e.g. node level targets of shape [num_nodes, *] or graph-level targets of shape [1,*]
        # pos: node position matrix with shape [num_nodes, num_dimensions]

        video_1_graph.validate(raise_on_error=True)
        return video_1_graph
        
    def __len__(self):
        return len(self.all_graphs)

    def __getitem__(self, idx):
        graph = self.all_graphs[idx]
        object_label = self.original_label[idx] # e.g. butter
        object_class_id = self.unique_label_mappings[object_label] # id corresponding to object e.g. 2
        labels = torch.tensor(object_class_id)
        return graph, labels # for gnn and cnn
