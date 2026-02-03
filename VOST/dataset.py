import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

class VODDataset(Dataset):
    def __init__(self, video_names, frames= (None,None), task='object_rec', split='train'):
        super().__init__()
        self.video_names = video_names
        self.init_frame = frames[0]
        self.final_frame = frames[1]            
        self.frames = []
        self.original_label = []
        self.unique_label_mappings = {}
        
        for video in self.video_names:
            parent_path = f"JPEGImages/{video}"

            frame_count = 0
            for f in os.listdir(parent_path):
                frame_count += 1
                if self.init_frame!=None and self.final_frame!=None:
                    if frame_count <= (self.final_frame - self.init_frame):
                        frame = cv2.imread(f"{parent_path}/{f}")
                        self.frames.append(frame) # 1080, 1920, 3

                        if task == 'object_rec':
                            target_name = video.split("_")[2] # object
                        else:
                            target_name = video.split("_")[1] # action
                        
                        self.original_label.append(target_name)
                        if target_name not in self.unique_label_mappings.keys(): # Ensure unique id for each object
                            self.unique_label_mappings[target_name] = self.video_names.index(video)
                else:
                    print("Extracting for all frames of a video")
       
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        img = torch.FloatTensor(self.frames[idx]).permute(2,0,1).float() if img.ndim == 3 else torch.FloatTensor(self.frames[idx]).permute(0,3,1,2).float()
        object_label = self.original_label[idx] # e.g. butter
        object_class_id = self.unique_label_mappings[object_label] # id corresponding to object e.g. 2
        labels = torch.tensor(object_class_id)
        return img, labels
