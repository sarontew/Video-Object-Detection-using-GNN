import torch.nn as nn
import torch
from torchvision import transforms, models

class CNN_Network(torch.nn.Module):
    def __init__(self, num_classes = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # with torch.no_grad():
        #     x = torch.zeros(500, 3, 32, 32) # determines correct input size for classification 
        #     features = self.backbone(x)
        #     #print(features.shape) # 1, 576
        #     self.feature_dimension = features.shape[1] # extract channel dimension of the feature map
        #     print("resnet feature dimension is")  
        #     print(self.feature_dimension)# 1000
        
        in_features = self.backbone.fc.in_features # 2048

        # self.layers = nn.Sequential(
        #     nn.Linear(self.feature_dimension, 256), #512
        #     nn.Hardswish(),
        #     nn.Dropout(0.3), #0.3
        #     nn.Linear(256, 1)
        # )
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # print("x is ", x)
        # print("x shape is", x.shape) # 1,3,1080
        # y = self.backbone(x)
        # z = self.layers(y)
        return self.backbone(x)

