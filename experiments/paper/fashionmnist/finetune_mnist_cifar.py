import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_mnist, load_cifar10, load_fashionmnist
from dlquantification.featureextraction.cnn import CNNFeatureExtractionModule, CNNFeatureExtractionModuleCifar10
import argparse
import numpy as np


def load_model(dataset, cuda_device):
    if dataset == "cifar10":
        fe = CNNFeatureExtractionModuleCifar10(output_size=256)
    elif dataset == "mnist":
        fe = CNNFeatureExtractionModule(output_size=256)
    elif dataset == "fashionmnist":
        fe = CNNFeatureExtractionModule(output_size=256)

    # Define the CNN
    class CNN(nn.Module):
        def __init__(self, feature_extraction):
            super(CNN, self).__init__()
            self.fe = feature_extraction
            self.fc2 = nn.Linear(self.fe.output_size, 10)

        def forward(self, x):
            # feature extraction module is prepare for batchs of bags, so we need to add a dimension
            return self.fc2(self.fe(x.unsqueeze(0)).squeeze(0))

    # Move model to device
    model = CNN(feature_extraction=fe).to(cuda_device)
    return model


def finetune_mnist_cifar10(dataset, cuda_device="cuda:0"):
    seed = 2032
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    # Hyperparameters
    num_epochs = 100
    learning_rate = 0.01
    batch_size = 128

    if dataset == "cifar10":
        dataset_train, dataset_val = load_cifar10(train=True, data_augmentation=False, seed=seed)
    elif dataset == "mnist":
        dataset_train, dataset_val = load_mnist(train=True, data_augmentation=False, seed=seed)
    elif dataset == "fashionmnist":
        dataset_train, dataset_val = load_fashionmnist(train=True, data_augmentation=False, seed=seed)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model = load_model(dataset, cuda_device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Implement early stopping
    patience = 5
    best_val_loss = float("inf")
    counter = 0

    # Train the model
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data and target to device
            data, target = data.to(cuda_device), target.to(cuda_device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}".format(
                        epoch + 1, num_epochs, batch_idx + 1, len(train_loader), loss.item()
                    )
                )
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                # Move data and target to device
                data, target = data.to(cuda_device), target.to(cuda_device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100 * correct / total
            print("Validation Loss: {:.4f}, Validation Accuracy: {} %".format(val_loss, val_acc))

            # Early stopping
            if val_loss < best_val_loss:
                counter = 0
                best_val_loss = val_loss
                torch.save(model.state_dict(), "savedmodels/finetune_model_{}.ckpt".format(dataset))
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping!")
                    break


def save_predictions(dataset, cuda_device="cuda:0"):
    seed = 2032
    if dataset == "cifar10":
        dataset_train, dataset_val = load_cifar10(train=True, data_augmentation=False, seed=seed)
        dataset_test = load_cifar10(train=False, data_augmentation=False, seed=seed)
    elif dataset == "mnist":
        dataset_train, dataset_val = load_mnist(train=True, data_augmentation=False, seed=seed)
        dataset_test = load_mnist(train=False, data_augmentation=False, seed=seed)
    elif dataset == "fashionmnist":
        dataset_train, dataset_val = load_fashionmnist(train=True, data_augmentation=False, seed=seed)
        dataset_test = load_fashionmnist(train=False, data_augmentation=False, seed=seed)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Load best model
    model = load_model(dataset, cuda_device)
    model.load_state_dict(torch.load("savedmodels/finetune_model_{}.ckpt".format(dataset)))

    # Test the model
    model.eval()
    with torch.no_grad():
        for loader_name, dataloader in zip(("train", "val", "test"), (train_loader, val_loader, test_loader)):
            correct = 0
            total = 0
            probabilities = []
            true = []
            for data, target in dataloader:
                # Move data and target to device
                data, target = data.to(cuda_device), target.to(cuda_device)
                output = model(data)
                prob = nn.functional.softmax(output, dim=1)
                probabilities.append(prob.cpu().numpy())
                true.append(target.cpu().numpy())
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            probabilities = np.concatenate(probabilities, axis=0)
            true = np.concatenate(true, axis=0)
            np.savetxt("predictions/output_predictions_{}_{}.csv".format(dataset, loader_name), probabilities)
            np.savetxt("predictions/true_{}_{}.csv".format(dataset, loader_name), true, fmt="%i")
            print("Test Accuracy of the model on the {} images: {} %".format(loader_name, 100 * correct / total))


if __name__ == "__main__":
    # Parametrice the script with argparse
    parser = argparse.ArgumentParser(description="MNIST/FASHIONMNIST/CIFAR finetuning script")
    parser.add_argument("-d", "--dataset", help="Dataset to use: MNIST, FASHIONMNIST, CIFAR10", required=True)
    parser.add_argument("-c", "--cuda_device", help="Device cuda:0 or cuda:1", required=True)
    print("Using following arguments:")
    args = vars(parser.parse_args())
    print(args)

    args["cuda_device"] = torch.device(args["cuda_device"])

    finetune_mnist_cifar10(**args)
    save_predictions(**args)

    # ------------> IMPORTANT NOTE: To be reprodubile this script needs to be launched as:
    # CUBLAS_WORKSPACE_CONFIG=:16:8 python finetune_mnist_cifar.py -d cifar10 -c cuda:0
