import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from const import class_labels

def masks_to_boxes(mask):
    boxes = []
    labels = []

    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0] ##
  
    for id in obj_ids:
        y, x = np.where(mask == id)
        if len(x) == 0:
            continue
        x1, y1, = x.min(), y.min()
        x2, y2 = x.max(), y.max()
        boxes.append((x1, y1, x2, y2))
        labels.append(int(id)) ## map to class label if needed
    return boxes, labels

class VODDataset(Dataset):
    def __init__(self, video_name, frames= (0,5)):
        super().__init__()
        self.video_name = video_name
        self.init_frame = frames[0]
        self.final_frame = frames[1]
        
        frames_vid_1 = []
        # add one more loop for each video
        parent_path = f"JPEGImages/{self.video_name}"
        for f in os.listdir(parent_path):
            frame = cv2.imread(f"{parent_path}/{f}")
            frames_vid_1.append(frame)
            
        self.annotations_vid_1 = []
        ## add one more loop for each video
        parent_path = f"Annotations/{self.video_name}"
        for f in os.listdir(parent_path):
            frame = cv2.imread(f"{parent_path}/{f}")
            self.annotations_vid_1.append(frame) # 1080, 1920, 3
        
        # Subset for quicker running
        self.annotations_vid_1 = self.annotations_vid_1[self.init_frame:self.final_frame]
            
        self.labels_tensor = []
        for i in range(len(self.annotations_vid_1)):
            rgb_mask = self.annotations_vid_1[i] # 1080,1920,3
            mask_flat = rgb_mask.reshape(-1, 3)
            colours, inverse = np.unique(mask_flat, axis=0, return_inverse = True)
            mask_id = inverse.reshape(rgb_mask.shape[:2])
            mask_id[(mask_flat  == 0).all(axis=1).reshape(rgb_mask.shape[:2])] = 0
            
            boxes, labels = masks_to_boxes(mask_id) # h,w
            self.labels_tensor.append(boxes)
       
    def __len__(self):
        return len(self.annotations_vid_1)

    def __getitem__(self, idx):
        img = self.annotations_vid_1[idx]
        img = torch.FloatTensor(self.annotations_vid_1[idx]).permute(2,0,1).float() if img.ndim == 3 else torch.FloatTensor(self.annotations_vid_1[idx]).permute(0,3,1,2).float()
        #b = self.labels_tensor[idx]
        #boxes = torch.FloatTensor(b) if len(b) == 1 else torch.FloatTensor(b[0])

        label = class_labels[self.video_name]
        
        labels = torch.tensor(label, dtype=torch.long) #"0_squeeze_cloth"
        
        # print(boxes.shape) #2,4
        # print(labels.shape) # 1
        
        # target = {
        #     "boxes": boxes, #nx4
        #     "labels": labels #n
        # }
        return img, labels
