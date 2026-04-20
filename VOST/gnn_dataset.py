import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
#from torch_geometric.data import Data, Dataset
from torchvision import transforms, models
from skimage.measure import label, regionprops
from torchvision.models.detection.roi_heads import roi_align
from PIL import Image
import torchvision.transforms.functional as F
import random
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),  # converts uint8 [0,255] -> float [0,1] and permutes channels
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def build_graph(node_features_per_frame):
    """
    Build a temporal graph from a list of frames.
    
    Args:
        node_features_per_frame: list of lists
            Each element is a list of torch tensors for the nodes in that frame.
            Example: [[f1_obj1, f1_obj2], [f2_obj1], [f3_obj1, f3_obj2, f3_obj3]]

    Returns:
        PyG Data object with node features and edge_index
    """

    nodes = []
    edges_from = []
    edges_to = []

    frame_offsets = {}  # keep track of node indices per frame
    offset = 0

    for i, frame_nodes in enumerate(node_features_per_frame):
        frame_offsets[i] = offset
        offset += len(frame_nodes)
        nodes.extend(frame_nodes)

    # Build temporal edges: connect every node in frame t → every node in frame t+1
    for t in range(len(node_features_per_frame) - 1):
        current_frame_nodes = node_features_per_frame[t]
        next_frame_nodes = node_features_per_frame[t + 1]

        for i in range(len(current_frame_nodes)):
            src = frame_offsets[t] + i
            for j in range(len(next_frame_nodes)):
                dst = frame_offsets[t + 1] + j
                edges_from.append(src)
                edges_to.append(dst)

    x = torch.stack(nodes)  # (num_nodes, feature_dim)
    edge_index = torch.tensor([edges_from, edges_to], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# frame1_nodes = [roi_align_feat1, roi_align_feat2]
# frame2_nodes = [roi_align_feat1]
# frame3_nodes = [roi_align_feat1, roi_align_feat2, roi_align_feat3]
# node_features_per_frame = [frame1_nodes, frame2_nodes, frame3_nodes]


import os
import cv2
import numpy as np
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class New_GNN_Dataset(Dataset):
    def __init__(self, video_names, split='train', task='object_rec', label_mapping=None):
        self.video_names = video_names
        self.split       = split
        self.task        = task

        if label_mapping is not None:
            self.unique_label_mappings = label_mapping
        else:
            self.unique_label_mappings = {}
            print('doing fixed framing')
            for video in video_names:
                target_name = (video.split("_")[2] if task == 'object_rec'
                               else video.split("_")[1])
                if target_name not in self.unique_label_mappings:
                    self.unique_label_mappings[target_name] = len(self.unique_label_mappings)

    # ── Joint spatial + colour augmentation ──────────────────────────────────

    def _augment(self, frame_np, mask_np):
        """
        frame_np : (H, W, 3)  uint8  BGR (as returned by cv2)
        mask_np  : (H, W)     uint8  binary {0, 1}

        Returns
        -------
        frame_t  : (3, H, W)  float32 tensor, ImageNet-normalised
        mask_t   : (H, W)     uint8  tensor, binary {0, 1}

        All *spatial* transforms are applied identically to both.
        Colour jitter is applied to the frame only.
        """
        # Convert frame BGR → RGB PIL image
        frame_pil = TF.to_pil_image(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
        # Mask as PIL image (mode "L") so TF functions work on it
        mask_pil  = TF.to_pil_image(mask_np)          # single-channel uint8

        # ── Shared spatial transforms ─────────────────────────────────────
        # 1. Random horizontal flip
        if random.random() < 0.5:
            frame_pil = TF.hflip(frame_pil)
            mask_pil  = TF.hflip(mask_pil)

        # 2. Random rotation (±10°)  — nearest for mask to avoid interpolation artefacts
        angle = random.uniform(-10, 10)
        frame_pil = TF.rotate(frame_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask_pil  = TF.rotate(mask_pil,  angle, interpolation=TF.InterpolationMode.NEAREST)

        # 3. Random resized crop (same params for both)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            frame_pil, scale=(0.8, 1.0), ratio=(3/4, 4/3)
        )
        frame_pil = TF.resized_crop(frame_pil, i, j, h, w,
                                     size=(224, 224),
                                     interpolation=TF.InterpolationMode.BILINEAR)
        mask_pil  = TF.resized_crop(mask_pil,  i, j, h, w,
                                     size=(224, 224),
                                     interpolation=TF.InterpolationMode.NEAREST)

        # ── Frame-only colour jitter ──────────────────────────────────────
        frame_pil = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        )(frame_pil)

        # ── Convert to tensors ────────────────────────────────────────────
        frame_t = TF.to_tensor(frame_pil)           # (3, 224, 224) float [0,1]
        frame_t = TF.normalize(frame_t,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        # Mask: PIL "L" → numpy → tensor; re-binarise to guard against interp bleed
        mask_t  = torch.from_numpy(
            (np.array(mask_pil) > 0).astype(np.uint8)
        )                                            # (224, 224) uint8

        return frame_t, mask_t

    def _val_transform(self, frame_np, mask_np):
        """Deterministic resize only — no augmentation."""
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        frame_pil = TF.to_pil_image(frame_rgb)
        mask_pil  = TF.to_pil_image(mask_np)

        frame_pil = TF.resize(frame_pil, (224, 224),
                               interpolation=TF.InterpolationMode.BILINEAR)
        mask_pil  = TF.resize(mask_pil,  (224, 224),
                               interpolation=TF.InterpolationMode.NEAREST)

        frame_t = TF.to_tensor(frame_pil)
        frame_t = TF.normalize(frame_t,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        mask_t  = torch.from_numpy(
            (np.array(mask_pil) > 0).astype(np.uint8)
        )
        return frame_t, mask_t

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video       = self.video_names[idx]
        frame_files = sorted(os.listdir(f"JPEGImages/{video}"))

        fixed_frame_size = 32

        if self.split == 'train':
            frame_files = frame_files[:len(frame_files) // 2]
        else:
            frame_files = frame_files[len(frame_files) // 2:]

        num_frames = len(frame_files)
        indices    = np.linspace(0, num_frames - 1, fixed_frame_size)
        indices    = np.round(indices).astype(int)
        frame_files = [frame_files[i] for i in indices]

        frames_out = []
        masks_out  = []

        for image_name in frame_files:
            frame = cv2.imread(f"JPEGImages/{video}/{image_name}")          # (H,W,3) BGR uint8
            mask_name = image_name[:-3] + "png"
            mask  = cv2.imread(f"Annotations/{video}/{mask_name}",
                               cv2.IMREAD_GRAYSCALE)                        # (H,W) uint8

            # Resize to a common base size before augmentation so crop params
            # are comparable regardless of original resolution
            frame = cv2.resize(frame, (256, 256))
            mask  = cv2.resize(mask,  (256, 256),
                               interpolation=cv2.INTER_NEAREST)
            binary_mask = (mask > 0).astype(np.uint8)

            if self.split == 'train':
                frame_t, mask_t = self._augment(frame, binary_mask)
            else:
                frame_t, mask_t = self._val_transform(frame, binary_mask)

            frames_out.append(frame_t)   # (3, 224, 224)
            masks_out.append(mask_t)     # (224, 224)

        # Stack: (F, 3, 224, 224) and (F, 224, 224)
        frames_tensor = torch.stack(frames_out)   # float32
        masks_tensor  = torch.stack(masks_out)    # uint8

        target_name = (video.split("_")[2] if self.task == 'object_rec'
                       else video.split("_")[1])
        label = torch.tensor(self.unique_label_mappings[target_name])

        return frames_tensor, masks_tensor, label

class New_GNN_Dataset2222(Dataset):
    def __init__(self, video_names, split='train', task='object_rec', label_mapping=None):
        self.video_names = video_names
        self.split = split
        self.task = task
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224))
        ])
        if label_mapping is not None:
            # Use the externally provided mapping
            self.unique_label_mappings = label_mapping
        else:
            self.unique_label_mappings = {}
            print('doing fixed framing')

            for video in video_names:
                target_name = video.split("_")[2] if task == 'object_rec' else video.split("_")[1]
                if target_name not in self.unique_label_mappings:
                    self.unique_label_mappings[target_name] = len(self.unique_label_mappings)

    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        video = self.video_names[idx]
        frame_files = sorted(os.listdir(f"JPEGImages/{video}"))

        fixed_frame_size = 32

        if self.split == 'train':
            frame_files = frame_files[:len(frame_files)//2]
        else:
            frame_files = frame_files[len(frame_files)//2:]

        # print(len(frame_files))

        stride = len(frame_files) // fixed_frame_size
        if stride == 0:
            stride = 1

        num_frames = len(frame_files)
        indices = np.linspace(0, num_frames - 1, fixed_frame_size)
        indices = np.round(indices).astype(int) # upsample if there are less than 32 frames

        frame_files = [frame_files[i] for i in indices]

        # frame_files = frame_files[::3]

        frames = []
        masks = []

        for image_name in frame_files:
            frame = cv2.imread(f"JPEGImages/{video}/{image_name}")
            # cv2.imshow("Frame", frame)
            # cv2.waitKey(0)
            mask_name = image_name[:-3] + "png"
            mask = cv2.imread(f"Annotations/{video}/{mask_name}", cv2.IMREAD_GRAYSCALE)

            frame = cv2.resize(frame, (224, 224))
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

            binary_mask = (mask > 0).astype(np.uint8)

            # Create colored mask (red)
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 2] = binary_mask * 255  # Red channel

            # Blend image and mask
            overlay = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

            # --- Apply mask ---
            # Expand mask to 3 channels
            binary_mask_3ch = np.repeat(binary_mask[:, :, None], 3, axis=2)

            # Multiply to keep only object
            masked_image = frame * binary_mask_3ch

            # Show result
            # cv2.imshow("Masked Object", masked_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            frames.append(frame)
            masks.append(binary_mask)

        #cv2.destroyAllWindows()

        frames = np.array(frames)
        masks = np.array(masks)

        # (F, H, W, C) → (F, C, H, W)
        #frames = frames.transpose(0, 3, 1, 2) / 255.0

        frames = torch.tensor(frames, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.long)

        target_name = video.split("_")[2] if self.task == 'object_rec' else video.split("_")[1]
        label = torch.tensor(self.unique_label_mappings[target_name])
        #print("in dataset frames and mask shape", frames.shape, masks.shape)

        return frames, masks, label

    # def __getitem__(self, idx):
    #     video = self.video_names[idx]

    #     frame_files = sorted(os.listdir(f"JPEGImages/{video}"))

    #     # Split frames
    #     if self.split == 'train':
    #         frame_files = frame_files[:len(frame_files)//2]
    #     else:
    #         frame_files = frame_files[len(frame_files)//2:]

    #     frames = []
    #     masks = []

    #     frame_files = frame_files[::3]

    #     for image_name in frame_files:
    #         frame = cv2.imread(f"JPEGImages/{video}/{image_name}")
    #         print("in gnn dataset frame shape", frame.shape)
            
    #         mask_name = image_name[:-3] + "png"
    #         mask = cv2.imread(f"Annotations/{video}/{mask_name}", cv2.IMREAD_GRAYSCALE)
    #         print("in gnn dataset mask shape", mask.shape)

    #         binary_mask = (mask > 0).astype(np.uint8) * 255

    #         frames.append(frame)
    #         masks.append(binary_mask)

    #     frames = np.array(frames)   # (F, H, W, C)
    #     masks = np.array(masks)     # (F, H, W)

    #     print("in dataset frames and mask shape", frames.shape, masks.shape)

    #     # label
    #     target_name = video.split("_")[2] if self.task == 'object_rec' else video.split("_")[1]
    #     label = torch.tensor(self.unique_label_mappings[target_name])

    #     return torch.tensor(frames), torch.tensor(masks), label



class Newold_GNN_Dataset(Dataset):

    # Instead of returning pooled features:

    # return graph, labels
    # return the raw information:

    # return {
    #     "frames": frames,
    #     "boxes": boxes,
    #     "label": label
    # }
    # Then during training:

    # features = backbone(frames)

    # nodes = roi_align(features, boxes)

    # graph = build_graph(nodes)

    # pred = gnn(graph)

    
    def __init__(self, video_names, split='train', task='object_rec'):
        
        self.all_target_labels = []
        self.unique_label_mappings = {}
        self.init_frame = None
        self.final_frame = None
        self.split = split
        self.video_names = video_names
        self.original_label = []
        self.regions_per_video = []
        self.frames_per_video = []
        self.binarymasks = []   
        print("okay new dataset")   

        for video in video_names:
            print(f"video {video}")
            # for one video
            frames = []
            fs = []
            regions = []
            labels = []
            frame_count = -1
            binary_masks = []

            for name in os.listdir(f"JPEGImages/{video}"):
                frames.append(name)

            for image_name in os.listdir(f"JPEGImages/{video}"):
                frame_count += 1
                
                if self.init_frame==None or self.final_frame==None:
                    # First half train, second half test
                    if self.split == 'train':
                        self.init_frame = 0
                        #self.final_frame = 4 ## temp
                        self.final_frame = len(frames) // 2
                    elif self.split == 'test':
                        self.init_frame = len(frames) // 2 + 1
                        self.final_frame = len(frames) -1
                        # self.init_frame = 5
                        # self.final_frame = 9

                if self.init_frame!=None and self.final_frame!=None:
                    if frame_count <= (self.final_frame - self.init_frame):
                        #print(" count", frame_count)

                        frame = cv2.imread(f"JPEGImages/{video}/{image_name}")
                        fs.append(frame)

                        mask_name = image_name[0:-3] + "png"

                        mask_parent_dir = "Annotations"
                        mask = cv2.imread(f"{mask_parent_dir}/{video}/{mask_name}", cv2.IMREAD_GRAYSCALE)
                        mask_gray = mask[:, :, 0] if len(mask.shape) == 3 else mask

                        # Robust threshold: anything > 0 is ROI
                        binary_mask = np.where(mask_gray > 0, 255, 0).astype(np.uint8) # 1080,1920
                        #print("in dataset, binary mask shape is", binary_mask.shape)
                        binary_masks.append(binary_mask)
                        roi = cv2.bitwise_and(frame, frame, mask=binary_mask)

                        labeled = label(binary_mask, connectivity=2) # 2 = 8 connectivity, type:ndarray
                        regions.append([regionprops(labeled)]) # regions per frame
                        #print("for this frame ", len(regionprops(labeled)))
                        #regions.append(binary_mask)
                
            
            self.regions_per_video.append(regions)
            #print("for this video the lenght of regions is ", len(regions))
            #self.frames_per_video.append(frames[self.init_frame:self.final_frame+1])
            self.frames_per_video.append(fs)
            #print("for this video the lenght of frames is ", len(frames[self.init_frame:self.final_frame+1]))
            #print("for this video the lenght of frames is ", len(fs))
            self.binarymasks.append(binary_masks)


            # Sort out label for each video
            if task == 'object_rec':
                target_name = video.split("_")[2] # object
            elif task == 'action_rec':
                target_name = video.split("_")[1] # action
            else:
                raise Exception("Valid task for object or action recognition not provided")
            self.original_label.append(target_name)

            if target_name not in self.unique_label_mappings.keys(): # Ensure unique id for each object
                self.unique_label_mappings[target_name] = self.video_names.index(video)

    
    def __len__(self):
        return len(self.frames_per_video)

    def __getitem__(self, idx):
        regions = self.regions_per_video[idx]
        frames = self.frames_per_video[idx]
        masks = self.binarymasks[idx]
        #print("frames", frames) # 1080,1920,3 type uint
        #print("regions", len(regions))
        object_label = self.original_label[idx] # e.g. butter
        object_class_id = self.unique_label_mappings[object_label] # id corresponding to object e.g. 2
        labels = torch.tensor(object_class_id)
        #print("labels", object_class_id)

        # frames = [transform(Image.open(p)) for p in frame_paths]
        # frames = torch.stack(frames)

        # region_boxes = []

        # for fregion in regions:          # per frame
        #     frame_boxes = []
        #     for region_list in fregion:  # list of regions
        #         for r in region_list:    # actual RegionProperties
        #             minr, minc, maxr, maxc = r.bbox
        #             frame_boxes.append([minr, minc, maxr, maxc])

        #     region_boxes.append(frame_boxes)
        # regions = torch.tensor(region_boxes, dtype=torch.float32)
            
        MAX_REGIONS = 10

        region_boxes = []

        for fregion in regions:
            frame_boxes = []

            for region_list in fregion:
                for r in region_list:
                    frame_boxes.append(list(r.bbox))

            # limit number of regions
            frame_boxes = frame_boxes[:MAX_REGIONS]

            # pad missing regions
            while len(frame_boxes) < MAX_REGIONS:
                frame_boxes.append([0,0,0,0])

            region_boxes.append(frame_boxes)

        regions_tensor = torch.tensor(region_boxes, dtype=torch.float32)
        
        frames = torch.tensor(np.array(frames))
        # frame = torch.from_numpy(frame).float()  # float32
        # frame = frame.permute(2,0,1) / 255.0     # normalize to [0,1] and channels first
        
        #return frames, regions_tensor, labels
        return frames, torch.tensor(np.array(masks)), labels

    

class GNN_Dataset(Dataset):

    def __init__(self, video_names, backbone, split='train', task='object_rec'):
        self.all_graphs = []
        self.all_frames = []
        self.all_target_labels = []
        self.unique_label_mappings = {}
        self.init_frame = None
        self.final_frame = None
        self.split = split
        self.video_names = video_names
        self.original_label = []
        self.backbone = backbone

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


    def generate_graph(self, video):
        # resnet = models.resnet50(pretrained=True)
        # backbone = torch.nn.Sequential(*list(resnet.children())[:-2])

        backbone = self.backbone 
        backbone.eval()
        frame_count = -1
        all_frames = []
        subimages_per_frame = {}

        print(f"creating graph for video {video}")

        for name in os.listdir(f"JPEGImages/{video}"):
            all_frames.append(name)

        for image_name in os.listdir(f"JPEGImages/{video}"):
            frame_count += 1

            if self.init_frame==None or self.final_frame==None:
                # First half train, second half test
                if self.split == 'train':
                    self.init_frame = 0
                    self.final_frame = 4
                    # self.final_frame = len(all_frames) // 2
                elif self.split == 'test':
                    # self.init_frame = len(all_frames) // 2 + 1
                    # self.final_frame = len(all_frames) -1
                    self.init_frame = 5
                    self.final_frame = 9

            if self.init_frame!=None and self.final_frame!=None:
                if frame_count <= (self.final_frame - self.init_frame):
                    print("frame count", frame_count)

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
                        with torch.no_grad(): # REMOVEEE
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

        print("there are number of nodes", len(nodes)) 

        #print("dataset subimages per frame is", subimages_per_frame.keys())

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
        print("labels", labels)
        print("graph", graph)
        return graph, labels # for gnn and cnn




# import os
# import cv2
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from torch_geometric.data import Data
# #from torch_geometric.data import Data, Dataset
# from torchvision import transforms, models
# from skimage.measure import label, regionprops
# from torchvision.models.detection.roi_heads import roi_align
# from PIL import Image
# import torchvision.transforms.functional as F
# import random
# import torchvision.transforms as T
# transform = T.Compose([
#     T.ToTensor(),  # converts uint8 [0,255] -> float [0,1] and permutes channels
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
# ])

# def build_graph(node_features_per_frame):
#     """
#     Build a temporal graph from a list of frames.
    
#     Args:
#         node_features_per_frame: list of lists
#             Each element is a list of torch tensors for the nodes in that frame.
#             Example: [[f1_obj1, f1_obj2], [f2_obj1], [f3_obj1, f3_obj2, f3_obj3]]
#     Returns:
#         PyG Data object with node features and edge_index
#     """
#     nodes = []
#     edges_from = []
#     edges_to = []
#     frame_offsets = {}  # keep track of node indices per frame
#     offset = 0
#     for i, frame_nodes in enumerate(node_features_per_frame):
#         frame_offsets[i] = offset
#         offset += len(frame_nodes)
#         nodes.extend(frame_nodes)
#     # Build temporal edges: connect every node in frame t → every node in frame t+1
#     for t in range(len(node_features_per_frame) - 1):
#         current_frame_nodes = node_features_per_frame[t]
#         next_frame_nodes = node_features_per_frame[t + 1]
#         for i in range(len(current_frame_nodes)):
#             src = frame_offsets[t] + i
#             for j in range(len(next_frame_nodes)):
#                 dst = frame_offsets[t + 1] + j
#                 edges_from.append(src)
#                 edges_to.append(dst)
#     x = torch.stack(nodes)  # (num_nodes, feature_dim)
#     edge_index = torch.tensor([edges_from, edges_to], dtype=torch.long)
#     return Data(x=x, edge_index=edge_index)

# # frame1_nodes = [roi_align_feat1, roi_align_feat2]
# # frame2_nodes = [roi_align_feat1]
# # frame3_nodes = [roi_align_feat1, roi_align_feat2, roi_align_feat3]
# # node_features_per_frame = [frame1_nodes, frame2_nodes, frame3_nodes]

# class New_GNN_Dataset(Dataset):
#     def __init__(self, video_names, split='train', task='object_rec'):
#         self.video_names = video_names
#         self.split = split
#         self.task = task
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize((224,224))
#         ])
#         self.unique_label_mappings = {}
#         for video in video_names:
#             target_name = video.split("_")[2] if task == 'object_rec' else video.split("_")[1]
#             if target_name not in self.unique_label_mappings:
#                 self.unique_label_mappings[target_name] = len(self.unique_label_mappings)
#         if self.split == 'train':
#             self.transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((256, 256)),
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomRotation(10),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#         else:
#             # no aug for valiation / test
#             self.transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#     def __len__(self):
#         return len(self.video_names)
    
#     def __getitem__(self, idx):
#         video = self.video_names[idx]
#         frame_files = sorted(os.listdir(f"JPEGImages/{video}"))
#         if self.split == 'train':
#             frame_files = frame_files[:len(frame_files)//2]
#         else:
#             frame_files = frame_files[len(frame_files)//2:]
#         frame_files = frame_files[::3]
#         frames = []
#         masks = []
#         for image_name in frame_files:
#             frame = cv2.imread(f"JPEGImages/{video}/{image_name}")
#             mask_name = image_name[:-3] + "png"
#             mask = cv2.imread(f"Annotations/{video}/{mask_name}", cv2.IMREAD_GRAYSCALE)
#             # ---- Resize first ----
#             frame = cv2.resize(frame, (256, 256))
#             mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
#             # ---- RANDOM CROP (shared) ----
#             h, w = frame.shape[:2]
#             new_h, new_w = 224, 224
#             top = np.random.randint(0, h - new_h + 1)
#             left = np.random.randint(0, w - new_w + 1)
#             frame = frame[top:top+new_h, left:left+new_w]
#             mask = mask[top:top+new_h, left:left+new_w]
#             # ---- RANDOM FLIP (shared) ----
#             if np.random.rand() > 0.5:
#                 frame = cv2.flip(frame, 1)
#                 mask = cv2.flip(mask, 1)
#             # ---- RANDOM ROTATION (shared) ----
#             angle = np.random.uniform(-10, 10)
#             M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
#             frame = cv2.warpAffine(frame, M, (224, 224), flags=cv2.INTER_LINEAR)
#             mask = cv2.warpAffine(mask, M, (224, 224), flags=cv2.INTER_NEAREST)
#             # ---- COLOR JITTER (ONLY IMAGE) ----
#             frame = frame.astype(np.float32) / 255.0
#             brightness = np.random.uniform(0.8, 1.2)
#             contrast = np.random.uniform(0.8, 1.2)
#             frame = np.clip(frame * contrast + brightness - 1, 0, 1)
#             # ---- NORMALIZE ----
#             mean = np.array([0.485, 0.456, 0.406])
#             std = np.array([0.229, 0.224, 0.225])
#             frame = (frame - mean) / std
#             # ---- BINARY MASK ----
#             binary_mask = (mask > 0).astype(np.uint8)
#             frames.append(frame)   # (H,W,C)
#             masks.append(binary_mask)  # (H,W)
#         frames = np.array(frames)   # (F,H,W,C)
#         masks = np.array(masks)     # (F,H,W)
#         frames = torch.tensor(frames, dtype=torch.float32)
#         masks = torch.tensor(masks, dtype=torch.long)
#         target_name = video.split("_")[2] if self.task == 'object_rec' else video.split("_")[1]
#         label = torch.tensor(self.unique_label_mappings[target_name])
#         return frames, masks, label
#     # def __getitem__(self, idx):
#     #     video = self.video_names[idx]
#     #     frame_files = sorted(os.listdir(f"JPEGImages/{video}"))
#     #     # Split frames
#     #     if self.split == 'train':
#     #         frame_files = frame_files[:len(frame_files)//2]
#     #     else:
#     #         frame_files = frame_files[len(frame_files)//2:]
#     #     frames = []
#     #     masks = []
#     #     frame_files = frame_files[::3]
#     #     for image_name in frame_files:
#     #         frame = cv2.imread(f"JPEGImages/{video}/{image_name}")
#     #         print("in gnn dataset frame shape", frame.shape)
            
#     #         mask_name = image_name[:-3] + "png"
#     #         mask = cv2.imread(f"Annotations/{video}/{mask_name}", cv2.IMREAD_GRAYSCALE)
#     #         print("in gnn dataset mask shape", mask.shape)
#     #         binary_mask = (mask > 0).astype(np.uint8) * 255
#     #         frames.append(frame)
#     #         masks.append(binary_mask)
#     #     frames = np.array(frames)   # (F, H, W, C)
#     #     masks = np.array(masks)     # (F, H, W)
#     #     print("in dataset frames and mask shape", frames.shape, masks.shape)
#     #     # label
#     #     target_name = video.split("_")[2] if self.task == 'object_rec' else video.split("_")[1]
#     #     label = torch.tensor(self.unique_label_mappings[target_name])
#     #     return torch.tensor(frames), torch.tensor(masks), label


# class Newold_GNN_Dataset(Dataset):
#     # Instead of returning pooled features:
#     # return graph, labels
#     # return the raw information:
#     # return {
#     #     "frames": frames,
#     #     "boxes": boxes,
#     #     "label": label
#     # }
#     # Then during training:
#     # features = backbone(frames)
#     # nodes = roi_align(features, boxes)
#     # graph = build_graph(nodes)
#     # pred = gnn(graph)
    
#     def __init__(self, video_names, split='train', task='object_rec'):
        
#         self.all_target_labels = []
#         self.unique_label_mappings = {}
#         self.init_frame = None
#         self.final_frame = None
#         self.split = split
#         self.video_names = video_names
#         self.original_label = []
#         self.regions_per_video = []
#         self.frames_per_video = []
#         self.binarymasks = []   
#         print("okay new dataset")   
#         for video in video_names:
#             print(f"video {video}")
#             # for one video
#             frames = []
#             fs = []
#             regions = []
#             labels = []
#             frame_count = -1
#             binary_masks = []
#             for name in os.listdir(f"JPEGImages/{video}"):
#                 frames.append(name)
#             for image_name in os.listdir(f"JPEGImages/{video}"):
#                 frame_count += 1
                
#                 if self.init_frame==None or self.final_frame==None:
#                     # First half train, second half test
#                     if self.split == 'train':
#                         self.init_frame = 0
#                         #self.final_frame = 4 ## temp
#                         self.final_frame = len(frames) // 2
#                     elif self.split == 'test':
#                         self.init_frame = len(frames) // 2 + 1
#                         self.final_frame = len(frames) -1
#                         # self.init_frame = 5
#                         # self.final_frame = 9
#                 if self.init_frame!=None and self.final_frame!=None:
#                     if frame_count <= (self.final_frame - self.init_frame):
#                         #print(" count", frame_count)
#                         frame = cv2.imread(f"JPEGImages/{video}/{image_name}")
#                         fs.append(frame)
#                         mask_name = image_name[0:-3] + "png"
#                         mask_parent_dir = "Annotations"
#                         mask = cv2.imread(f"{mask_parent_dir}/{video}/{mask_name}", cv2.IMREAD_GRAYSCALE)
#                         mask_gray = mask[:, :, 0] if len(mask.shape) == 3 else mask
#                         # Robust threshold: anything > 0 is ROI
#                         binary_mask = np.where(mask_gray > 0, 255, 0).astype(np.uint8) # 1080,1920
#                         #print("in dataset, binary mask shape is", binary_mask.shape)
#                         binary_masks.append(binary_mask)
#                         roi = cv2.bitwise_and(frame, frame, mask=binary_mask)
#                         labeled = label(binary_mask, connectivity=2) # 2 = 8 connectivity, type:ndarray
#                         regions.append([regionprops(labeled)]) # regions per frame
#                         #print("for this frame ", len(regionprops(labeled)))
#                         #regions.append(binary_mask)
                
            
#             self.regions_per_video.append(regions)
#             #print("for this video the lenght of regions is ", len(regions))
#             #self.frames_per_video.append(frames[self.init_frame:self.final_frame+1])
#             self.frames_per_video.append(fs)
#             #print("for this video the lenght of frames is ", len(frames[self.init_frame:self.final_frame+1]))
#             #print("for this video the lenght of frames is ", len(fs))
#             self.binarymasks.append(binary_masks)

#             # Sort out label for each video
#             if task == 'object_rec':
#                 target_name = video.split("_")[2] # object
#             elif task == 'action_rec':
#                 target_name = video.split("_")[1] # action
#             else:
#                 raise Exception("Valid task for object or action recognition not provided")
#             self.original_label.append(target_name)
#             if target_name not in self.unique_label_mappings.keys(): # Ensure unique id for each object
#                 self.unique_label_mappings[target_name] = self.video_names.index(video)
    
#     def __len__(self):
#         return len(self.frames_per_video)
#     def __getitem__(self, idx):
#         regions = self.regions_per_video[idx]
#         frames = self.frames_per_video[idx]
#         masks = self.binarymasks[idx]
#         #print("frames", frames) # 1080,1920,3 type uint
#         #print("regions", len(regions))
#         object_label = self.original_label[idx] # e.g. butter
#         object_class_id = self.unique_label_mappings[object_label] # id corresponding to object e.g. 2
#         labels = torch.tensor(object_class_id)
#         #print("labels", object_class_id)
#         # frames = [transform(Image.open(p)) for p in frame_paths]
#         # frames = torch.stack(frames)
#         # region_boxes = []
#         # for fregion in regions:          # per frame
#         #     frame_boxes = []
#         #     for region_list in fregion:  # list of regions
#         #         for r in region_list:    # actual RegionProperties
#         #             minr, minc, maxr, maxc = r.bbox
#         #             frame_boxes.append([minr, minc, maxr, maxc])
#         #     region_boxes.append(frame_boxes)
#         # regions = torch.tensor(region_boxes, dtype=torch.float32)
            
#         MAX_REGIONS = 10
#         region_boxes = []
#         for fregion in regions:
#             frame_boxes = []
#             for region_list in fregion:
#                 for r in region_list:
#                     frame_boxes.append(list(r.bbox))
#             # limit number of regions
#             frame_boxes = frame_boxes[:MAX_REGIONS]
#             # pad missing regions
#             while len(frame_boxes) < MAX_REGIONS:
#                 frame_boxes.append([0,0,0,0])
#             region_boxes.append(frame_boxes)
#         regions_tensor = torch.tensor(region_boxes, dtype=torch.float32)
        
#         frames = torch.tensor(np.array(frames))
#         # frame = torch.from_numpy(frame).float()  # float32
#         # frame = frame.permute(2,0,1) / 255.0     # normalize to [0,1] and channels first
        
#         #return frames, regions_tensor, labels
#         return frames, torch.tensor(np.array(masks)), labels
    
# class GNN_Dataset(Dataset):
#     def __init__(self, video_names, backbone, split='train', task='object_rec'):
#         self.all_graphs = []
#         self.all_frames = []
#         self.all_target_labels = []
#         self.unique_label_mappings = {}
#         self.init_frame = None
#         self.final_frame = None
#         self.split = split
#         self.video_names = video_names
#         self.original_label = []
#         self.backbone = backbone
#         for video in video_names:
#             graph = self.generate_graph(video)
#             self.all_graphs.append(graph)
#             if task == 'object_rec':
#                 target_name = video.split("_")[2] # object
#             elif task == 'action_rec':
#                 target_name = video.split("_")[1] # action
#             else:
#                 raise Exception("Valid task for object or action recognition not provided")
            
#             self.original_label.append(target_name)
#             if target_name not in self.unique_label_mappings.keys(): # Ensure unique id for each object
#                 self.unique_label_mappings[target_name] = self.video_names.index(video)

#     def generate_graph(self, video):
#         # resnet = models.resnet50(pretrained=True)
#         # backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
#         backbone = self.backbone 
#         backbone.eval()
#         frame_count = -1
#         all_frames = []
#         subimages_per_frame = {}
#         print(f"creating graph for video {video}")
#         for name in os.listdir(f"JPEGImages/{video}"):
#             all_frames.append(name)
#         for image_name in os.listdir(f"JPEGImages/{video}"):
#             frame_count += 1
#             if self.init_frame==None or self.final_frame==None:
#                 # First half train, second half test
#                 if self.split == 'train':
#                     self.init_frame = 0
#                     self.final_frame = 4
#                     # self.final_frame = len(all_frames) // 2
#                 elif self.split == 'test':
#                     # self.init_frame = len(all_frames) // 2 + 1
#                     # self.final_frame = len(all_frames) -1
#                     self.init_frame = 5
#                     self.final_frame = 9
#             if self.init_frame!=None and self.final_frame!=None:
#                 if frame_count <= (self.final_frame - self.init_frame):
#                     print("frame count", frame_count)
#                     frame = cv2.imread(f"JPEGImages/{video}/{image_name}")
#                     subimages_per_frame[frame_count] = []
#                     mask_name = image_name[0:-3] + "png"
#                     mask_parent_dir = "Annotations"
#                     mask = cv2.imread(f"{mask_parent_dir}/{video}/{mask_name}", cv2.IMREAD_GRAYSCALE)
#                     mask_gray = mask[:, :, 0] if len(mask.shape) == 3 else mask
#                     # Robust threshold: anything > 0 is ROI
#                     binary_mask = np.where(mask_gray > 0, 255, 0).astype(np.uint8)
#                     roi = cv2.bitwise_and(frame, frame, mask=binary_mask)
#                     labeled = label(binary_mask, connectivity=2) # 2 = 8 connectivity, type:ndarray
#                     regions = regionprops(labeled) # type: list
#                     # Get each region's sub-image from masked out
#                     for r in regions:
#                         min_row, min_col, max_row, max_col = r.bbox
#                         subimage = roi[min_row:max_row, min_col:max_col]
#                         frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float()
#                         with torch.no_grad(): # REMOVEEE
#                             backbone_features = backbone(frame_tensor)
#                             bbox = torch.tensor([[0, min_col, min_row, max_col, max_row]], dtype=torch.float32)
                            
#                             pooled = roi_align(input=backbone_features, boxes=bbox, output_size=(7,7)) # 1, 2048, 7, 7
#                             pooled = pooled.squeeze(0) # remove the first dim
#                             #subimages_per_frame[frame_count].append(subimage) # each chunk from each frame
#                             # add the pooled features instead
#                             subimages_per_frame[frame_count].append(pooled.cpu())
#                     # Graph creation
#                     # Node: each sub-images from every single frame
#                     # Edge: connecting subimages from frame i to every subimage from frame i+1
#                     edges_connecting_from = []
#                     edges_connecting_to = []
#                     sub_images_as_nodes = []
#                     # Fix frame offset for the edge node count
#                     frame_ids = sorted(subimages_per_frame.keys())
#                     frame_offsets = {}
#                     offset = 0
#                     for frame_id in frame_ids:
#                         frame_offsets[frame_id] = offset
#                         offset += len(subimages_per_frame[frame_id])
#                     # Create nodes and edges
#                     for idx, frame_id in enumerate(frame_ids):
#                         sub_images = subimages_per_frame[frame_id]
#                         sub_images_as_nodes.extend(sub_images) # flat nodes
#                         if idx == len(frame_ids) - 1: # skip last frame
#                             continue
#                         next_frame_id = frame_ids[idx + 1]
#                         next_frame_subimages = subimages_per_frame[next_frame_id]
#                         for current_subimage_index in range(len(sub_images)):
#                             source_node = frame_offsets[frame_id] + current_subimage_index
#                             for next_frame_index in range(len(next_frame_subimages)): # create an edge for all of next frame's subimages
#                                 target_node = frame_offsets[next_frame_id] + next_frame_index
#                                 edges_connecting_from.append(source_node)
#                                 edges_connecting_to.append(target_node)
#         all_nodes = []
#         for frame_id in subimages_per_frame:
#             for pooled in subimages_per_frame[frame_id]:
#                 all_nodes.append(pooled)
#         nodes = torch.stack(all_nodes)
#         print("there are number of nodes", len(nodes)) 
#         #print("dataset subimages per frame is", subimages_per_frame.keys())
#         video_1_graph = Data(x=nodes, edge_index=torch.tensor([edges_connecting_from, edges_connecting_to], dtype=torch.long))
#         # x: node feature matrix with shape [num_nodes, num_node_features]
#         # edge_index: graph connectivity in COO format with shape [2, num_edges]
#         # edge_attr: edge feature matrix with shape [num_edge, num_edge_features]
#         # y: target to train against e.g. node level targets of shape [num_nodes, *] or graph-level targets of shape [1,*]
#         # pos: node position matrix with shape [num_nodes, num_dimensions]
#         video_1_graph.validate(raise_on_error=True)
#         return video_1_graph
        
#     def __len__(self):
#         return len(self.all_graphs)
#     def __getitem__(self, idx):
#         graph = self.all_graphs[idx]
#         object_label = self.original_label[idx] # e.g. butter
#         object_class_id = self.unique_label_mappings[object_label] # id corresponding to object e.g. 2
#         labels = torch.tensor(object_class_id)
#         print("labels", labels)
#         print("graph", graph)
#         return graph, labels # for gnn and cnn
