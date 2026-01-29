import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from const import class_labels

class VODDataset(Dataset):
    def __init__(self, video_names, frames= (None,None)):
        super().__init__()
        self.video_names = video_names
        self.init_frame = frames[0]
        self.final_frame = frames[1]            
        self.annotations_vid_1 = []
        self.labels = []
        
        for video in self.video_names:
            parent_path = f"Annotations/{video}"
            frame_count = 0
            for f in os.listdir(parent_path):
                frame_count += 1
                if self.init_frame!=None and self.final_frame!=None:
                    if frame_count <= (self.final_frame - self.init_frame):
                        frame = cv2.imread(f"{parent_path}/{f}")
                        self.annotations_vid_1.append(frame) # 1080, 1920, 3
                        self.labels.append(video)
                else:
                    print("Extracting for all frames of a video")
       
    def __len__(self):
        return len(self.annotations_vid_1)

    def __getitem__(self, idx):
        img = self.annotations_vid_1[idx]
        img = torch.FloatTensor(self.annotations_vid_1[idx]).permute(2,0,1).float() if img.ndim == 3 else torch.FloatTensor(self.annotations_vid_1[idx]).permute(0,3,1,2).float()
        video_label = self.labels[idx]
        label = class_labels[video_label]
        labels = torch.tensor(label, dtype=torch.long) # Class id mapped to video name
        return img, labels
