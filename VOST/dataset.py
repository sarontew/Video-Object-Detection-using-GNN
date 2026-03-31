import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from torchvision import transforms


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
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((224,224))
        # ])
        self.video_ids = []
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # no aug for valiation / test
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        subfolder = split
        
        for video in self.video_names:
            parent_path = f"JPEGImages/{video}"
            print("extracting frames for video", video)
            all_frames = sorted(os.listdir(parent_path))

            if task == 'object_rec':
                #target_name = video.split("_")[2] # object
                parts = video.split("_")
                target_name = "_".join(parts[2:])
            elif task == 'action_rec':
                target_name = video.split("_")[1] # action
            else:
                raise Exception("Valid task for object or action recognition not provided")

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
            #for f in os.listdir(parent_path):
            for f in sorted(os.listdir(parent_path)):
                frame_count += 1
                if self.init_frame!=None and self.final_frame!=None:
                    if frame_count <= (self.final_frame - self.init_frame): # and frame_count % 3 == 0 (sample every 3rd frame)?

                        video_id = self.video_names.index(video)
                        self.video_ids.append(video)  # video name = group
                        frame_id = frame_count

                        frame_path = f"Resnet50_Features/{subfolder}/video{video_id}_frame_{frame_id}_resnet50_{split}_features.pt"
                        if os.path.exists(frame_path):
                            #print("path exists", frame_path)
                            continue

                        #print("path doesnt exist", frame_path)     
                        frame = cv2.imread(f"{parent_path}/{f}")
                        self.frames.append(frame) # 1080, 1920, 3
                        
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
        img = self.transform(img)
        #img = torch.FloatTensor(self.frames[idx]).permute(2,0,1).float() if img.ndim == 3 else torch.FloatTensor(self.frames[idx]).permute(0,3,1,2).float()
        object_label = self.original_label[idx] # e.g. butter
        # object_class_id = self.unique_label_mappings[object_label] # id corresponding to object e.g. 2
        # labels = torch.tensor(object_class_id)
        # return img, labels
    
        # Extract label from filename
        #filename = self.video_names[idx]
        #object_label = extract_label_from_filename(filename)
        
        object_class_id = self.unique_labels_map[object_label]
        labels = torch.tensor(object_class_id)
        video_id = self.video_ids[idx]
        
        return img, labels


import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class OtherVODDataset(Dataset):
    def __init__(self, video_names, unique_label_mappings, task='object_rec', split='train'):
        super().__init__()
        self.video_names = video_names       
        self.frames = []
        self.original_label = []
        self.unique_label_mappings = {}
        self.split = split
        self.unique_labels_map = unique_label_mappings
        self.video_ids = []
        self.frame_names = []

        
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # no aug for valiation / test
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        for video in self.video_names:    
            parent_path = f"JPEGImages/{video}"
            print("extracting frames for video", video)
            all_frames = sorted(os.listdir(parent_path))

            if task == 'object_rec':
                parts = video.split("_")
                target_name = "_".join(parts[2:])
            elif task == 'action_rec':
                target_name = video.split("_")[1]
            else:
                raise Exception("Valid task not provided")

            if self.split == 'train':
                self.init_frame = 0
                self.final_frame = len(all_frames) // 2
            elif self.split == 'test':
                self.init_frame = len(all_frames) // 2 + 1
                self.final_frame = len(all_frames) - 1

            frame_count = 0
            sample_freq = 3

            for f in os.listdir(parent_path):
                frame_count += 1
                if self.init_frame!=None and self.final_frame!=None:
                    if frame_count <= (self.final_frame - self.init_frame) and (frame_count % sample_freq == 0): # sample every 3rd

                        video_id = self.video_names.index(video)
                        self.video_ids.append(video)  # video name = group
                        frame_id = frame_count

                        #print("path doesnt exist", frame_path)     
                        # frame = cv2.imread(f"{parent_path}/{f}")
                        # self.frames.append(frame) # 1080, 1920, 3
                        self.frame_names.append(f"{parent_path}/{f}")
                        
                        self.original_label.append(target_name)
                        if target_name not in self.unique_label_mappings.keys(): # Ensure unique id for each object
                            self.unique_label_mappings[target_name] = self.video_names.index(video)
                            # print("target name is", target_name)
                            # print("unique label id", self.video_names.index(video))
                            # print("video id is", video)
                else:
                    print("Extracting for all frames of a video")
       
    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx):
        frame_name = self.frame_names[idx]

        # Read with cv2 and convert BGR → RGB
        img = cv2.imread(frame_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        object_label = self.original_label[idx]
        object_class_id = self.unique_labels_map[object_label]
        labels = torch.tensor(object_class_id)

        video_id = self.video_ids[idx]

        return img, labels, video_id