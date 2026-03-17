import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet50

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_file_names(filename):
    files =  []
    with open(f'ImageSets/{filename}', 'r') as fh:
        for line in fh:
            file = line.replace('\n', '')
            files.append((file))
    return files

def get_unique_labels(train_files, test_files, task="object_rec"):
    """
    Docstring for get_unique_labels
    
    :param train_files: files names used for training
    :param test_files: files names used for testing
    :param task: object or action recognition

    Returns the total unique labels for both the training and testing classes for the classifier
    Ensure duplicate classes within training and testing are not counted twice
    """
    total_labels = []
    def _get_one_file(files):
        labels = []
        for file in files:
            if task == "object_rec":
                label = file.split("_")[2] # object
            elif task == "action_rec":
                label = file.split("_")[1] # action
            labels.append(label)
        return labels
    
    total_labels = _get_one_file(train_files) + _get_one_file(test_files)
    return list(set(total_labels))


def save_features(data_loader, filename, train=True, aggregate=False):
    """ 
        Input: 
            data_loader : frames from videos
            filename : feature file name for saving
    
        Passes frames through resnet50. Then aggregates frames (mean) and stores the video features alongside the label ID
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    final_labels = []
    final_features = []
    video_to_frame_features = {} # video_label : [frame_feat1, frame_feat2]
    subfolder = 'train' if train else 'test'
    for images, labels in data_loader:

        images = images.to(device)

        with torch.inference_mode():
            features = model(images)

        features = features.cpu()

        for feature, label in zip(features, labels):

            label = label.item()

            if label not in video_to_frame_features:
                video_to_frame_features[label] = []

            video_to_frame_features[label].append(feature)

            final_labels.append(label)
            final_features.append(feature)

    for k in video_to_frame_features.keys():
        # print("saving per video for video with key", k)
        # print("frames:", len(video_to_frame_features[k])) # length of frames printed
        print("saving to")
        print(f"Resnet50_Features/{subfolder}/video{k}")
        for f in range(len(video_to_frame_features[k])):
            torch.save({
                "features": video_to_frame_features[k][f].cpu(), # number of vids x 2048
                "labels": k
            }, f"Resnet50_Features/{subfolder}/video{k}_frame_{f}_{filename}")

            # E.g. Resnet50_Features/train/video0_frame_4_resnet50_train_features.pt


    if aggregate:
        aggregated_video_features = []
        aggregated_video_labels=[]
        
        for label, frames in video_to_frame_features.items():
            averaged_features = np.mean(frames, axis=0) # aggregated frame features
            aggregated_video_features.append(torch.from_numpy(averaged_features))
            aggregated_video_labels.append(torch.tensor([label]))
        
        final_labels = aggregated_video_labels
        final_features = aggregated_video_features
        
        torch.save({
            "features": torch.cat(final_features), # number of vids x 2048
            "labels": torch.cat(final_labels)
        }, filename)

# def save_region_features(subimage, filename, data_loader):
def save_region_features(data_loader, filename="7_squeeze_pasta_graph_features.pt"):
    feature_nodes = []
    features_batch = []
    edge_index = []

    model = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval()

    for batch in data_loader:
        batch = batch.to(device)
        for node in batch.x:
            node = node.to(device)
            #print(node.numpy().shape)
            cv2.imshow("subimage", node.numpy())
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #     with torch.no_grad():
        #         node_features = model(node)
        #     feature_nodes.append(node_features)
        # edge_index = batch.edge_index
        # features_batch = np.stack(node_features)
    
    # print("node features shape", torch.cat(feature_nodes).shape)

    # torch.save({
    #     "x": torch.cat(feature_nodes), # number of vids x 2048
    #     "edge_index": torch.cat(edge_index),
    #     "batch": torch.cat(features_batch)
    # }, filename )