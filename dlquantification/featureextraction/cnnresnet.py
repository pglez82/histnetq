import torch.nn as nn
import torchvision
import torch


class CNNResnetFeatureExtractionModule(nn.Module):
    def __init__(self, output_size, pretrained=True, extra_linear_size=None, dropout=0):
        super(CNNResnetFeatureExtractionModule, self).__init__()

        self.extra_linear_size = extra_linear_size
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)
        if extra_linear_size is not None:
            self.out = torch.nn.Linear(output_size, extra_linear_size)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(p=dropout)
            self.output_size = extra_linear_size
        else:
            self.output_size = output_size

    def forward(self, x):
        batch_size = x.shape[0]
        bag_size = x.shape[1]
        # We process all the examples in all the samples at the same time
        x = x.view(bag_size * batch_size, *x.shape[2:])
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = x.view(batch_size, bag_size, -1)
        if self.extra_linear_size is None:
            return x
        else:
            return self.out(self.dropout(self.relu(x)))


class CNNFinetunedResnetFeatureExtractionModule(nn.Module):
    """
    Pretrained CNN used like a classifier
    """

    def __init__(self, n_classes, model_path, train_resnet: bool = True, extra_linear_size=None, dropout=0):
        super(CNNFinetunedResnetFeatureExtractionModule, self).__init__()

        self.extra_linear_size = extra_linear_size
        self.resnet = torchvision.models.resnet18()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)
        state_dict = torch.load(model_path)
        # We need to remove the prefix because it was saved with dataparallel
        state_dict = {k.partition("module.")[2]: v for k, v in state_dict.items()}
        self.resnet.load_state_dict(state_dict)
        self.set_train_resnet(train_resnet)
        if extra_linear_size is not None:
            self.out = torch.nn.Linear(n_classes, extra_linear_size)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(p=dropout)
            self.output_size = extra_linear_size
        else:
            self.output_size = n_classes

    def set_train_resnet(self, train_resnet):
        self.train_resnet = train_resnet
        for _, param in self.resnet.named_parameters():
            param.requires_grad = train_resnet

    def forward(self, x):
        batch_size = x.shape[0]
        bag_size = x.shape[1]
        # We process all the examples in all the samples at the same time
        x = x.view(bag_size * batch_size, *x.shape[2:])
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = x.view(batch_size, bag_size, -1)
        if self.extra_linear_size is None:
            return x
        else:
            return self.out(self.dropout(self.relu(x)))
