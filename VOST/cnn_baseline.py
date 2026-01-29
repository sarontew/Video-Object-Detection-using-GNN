import torch.nn as nn
import torch
from torchvision import transforms, models

class CNN_Classifier(torch.nn.Module):
    def __init__(self, num_classes = 30, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.classifier = nn.Sequential( #MLP head, more parameters, higher accuracy
            nn.Linear(2048, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x):
        return self.classifier(x)


class CNN_With_Backbone(torch.nn.Module):
    def __init__(self, num_classes = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        in_features = self.backbone.fc.in_features # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        ) 
    def forward(self, x):
        return self.backbone(x)

