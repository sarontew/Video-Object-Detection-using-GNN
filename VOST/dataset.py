import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

def extract_label_from_filename(filename):
    # Split by underscore, take last part before extension
    return filename.split('_')[-1].split('.')[0]  # 'cloth' in your example

class VODDataset(Dataset):
    def __init__(self, video_names, unique_label_mappings, frames= (None,None), task='object_rec', split='train'):
        """
        :param video_names: Video names to extract frames from
        :param frames: Range of frames to read
        :param task: 'object_rec' by default otherwise 'action_rec'
        :param split: train, test or val
        """
        super().__init__()
        self.video_names = video_names
        self.init_frame = frames[0]
        self.final_frame = frames[1]            
        self.frames = []
        self.original_label = []
        self.unique_label_mappings = {}
        self.split = split
        self.unique_labels_map = unique_label_mappings 
        
        for video in self.video_names:
            parent_path = f"JPEGImages/{video}"
            print("extracting frames for video", video)
            all_frames = []
            for name in os.listdir(parent_path):
                all_frames.append(name)

            if self.init_frame==None or self.final_frame==None:
                # First half train, second half test
                if self.split == 'train':
                    self.init_frame = 0
                    self.final_frame = len(all_frames) // 2
                elif self.split == 'test':
                    self.init_frame = len(all_frames) // 2 + 1
                    self.final_frame = len(all_frames) -1

                # Second half training, first half testing
                # if self.split == 'train':
                #     self.init_frame = len(all_frames) // 2 + 1
                #     self.final_frame = len(all_frames) -1
                # elif self.split == 'test':
                #     self.init_frame = 0
                #     self.final_frame = len(all_frames) // 2

                # Todo: First half training, both first half and second half testing (and vice versa)

            frame_count = 0
            for f in os.listdir(parent_path):
                frame_count += 1
                if self.init_frame!=None and self.final_frame!=None:
                    if frame_count <= (self.final_frame - self.init_frame):
                        frame = cv2.imread(f"{parent_path}/{f}")
                        self.frames.append(frame) # 1080, 1920, 3

                        if task == 'object_rec':
                            target_name = video.split("_")[2] # object
                        elif task == 'action_rec':
                            target_name = video.split("_")[1] # action
                        else:
                            raise Exception("Valid task for object or action recognition not provided")
                        
                        self.original_label.append(target_name)
                        if target_name not in self.unique_label_mappings.keys(): # Ensure unique id for each object
                            self.unique_label_mappings[target_name] = self.video_names.index(video)
                            # print("target name is", target_name)
                            # print("unique label id", self.video_names.index(video))
                            # print("video id is", video)
                else:
                    print("Extracting for all frames of a video")
       
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        img = torch.FloatTensor(self.frames[idx]).permute(2,0,1).float() if img.ndim == 3 else torch.FloatTensor(self.frames[idx]).permute(0,3,1,2).float()
        object_label = self.original_label[idx] # e.g. butter
        # object_class_id = self.unique_label_mappings[object_label] # id corresponding to object e.g. 2
        # labels = torch.tensor(object_class_id)
        # return img, labels
    
        # Extract label from filename
        #filename = self.video_names[idx]
        #object_label = extract_label_from_filename(filename)
        
        object_class_id = self.unique_labels_map[object_label]
        labels = torch.tensor(object_class_id)
        
        return img, labels
