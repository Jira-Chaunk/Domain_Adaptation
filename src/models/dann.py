import torch
import torch.nn as nn
import torch.optim as optim

# Define the Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(FeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

# Define the Task Classifier
class TaskClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(TaskClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Define the Domain Classifier
class DomainClassifier(nn.Module):
    def __init__(self, feature_dim):
        super(DomainClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
