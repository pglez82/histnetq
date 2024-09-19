import pandas as pd
import numpy as np
import torch
from dlquantification.transnet import TransNet
from dlquantification.histnet import HistNet
from dlquantification.deepsets import DeepSets
from dlquantification.utils.utils import APPBagGenerator
from torch.utils.data.dataset import TensorDataset
from dlquantification.featureextraction.cnn import CNNFeatureExtractionModule, CNNFeatureExtractionModuleCifar10
import os
from dlquantification.utils.lossfunc import MRAE
import json
import argparse
from tqdm import tqdm
from utils import load_cifar10, load_mnist, load_fashionmnist
from collections import OrderedDict


def train_mnist_cifar10(
    train_name, network, network_parameters, loss_func, dataset, load_weights, cuda_device="cuda:0"
):
    seed = 2032

    if dataset == "cifar10":
        dataset_train, dataset_val = load_cifar10(train=True, data_augmentation=False, seed=seed)
        common_param_path = "../parameters/common_parameters_cifar10.json"
    elif dataset == "mnist":
        dataset_train, dataset_val = load_mnist(train=True, data_augmentation=False, seed=seed)
        common_param_path = "../parameters/common_parameters_mnist.json"
    elif dataset == "fashionmnist":
        dataset_train, dataset_val = load_fashionmnist(train=True, data_augmentation=False, seed=seed)
        common_param_path = "../parameters/common_parameters_fashionmnist.json"

    with open(common_param_path, "r") as f:
        common_parameters = json.loads(f.read())
    with open(network_parameters, "r") as f:
        network_parameters = json.loads(f.read())

    # Bag generators
    train_bag_generator = APPBagGenerator(device=cuda_device, seed=seed)
    val_bag_generator = APPBagGenerator(device=cuda_device, seed=seed)

    # Loss function
    n_classes = 10
    loss_mrae = MRAE(eps=1.0 / (2 * common_parameters["bag_size"]), n_classes=n_classes).MRAE

    torch.manual_seed(seed)

    if dataset == "mnist":
        fe = CNNFeatureExtractionModule(output_size=256)
    elif dataset == "fashionmnist":
        fe = CNNFeatureExtractionModule(output_size=256)
    elif dataset == "cifar10":
        fe = CNNFeatureExtractionModuleCifar10(output_size=256)

    if load_weights:
        # LOAD finetuned model for classification
        weigths_to_load = OrderedDict()
        for key, value in torch.load(
            "savedmodels/finetune_model_{}.ckpt".format(dataset), map_location=torch.device("cpu")
        ).items():
            if key.startswith("fe."):
                weigths_to_load[key[3:]] = value
        fe.load_state_dict(weigths_to_load, strict=True)

    parameters = {**common_parameters, **network_parameters}
    parameters["n_classes"] = n_classes
    parameters["random_seed"] = seed
    parameters["feature_extraction_module"] = fe
    parameters["bag_generator"] = train_bag_generator
    parameters["val_bag_generator"] = val_bag_generator
    parameters["device"] = cuda_device
    if loss_func == "ae":
        parameters["quant_loss"] = torch.nn.L1Loss()
    elif loss_func == "mse":
        parameters["quant_loss"] = torch.nn.MSELoss()
    elif loss_func == "rae":
        parameters["quant_loss"] = loss_mrae
    else:
        raise ValueError("Loss function should be one of ae, mse or rae")
    # parameters["tensorboard_dir"] = "runs/" + dataset
    parameters["save_model_path"] = "savedmodels/" + train_name + ".pkl"
    parameters["wandb_experiment_name"] = train_name
    parameters["use_wandb"] = False
    print("Network parameters: ", parameters)

    if network == "histnet":
        model = HistNet(**parameters)
    elif network == "settransformers":
        model = TransNet(**parameters)
    elif network == "deepsets":
        model = DeepSets(**parameters)
    else:
        raise ValueError("network has not a proper value")

    model.fit(dataset=dataset_train, val_dataset=dataset_val)

    return model, loss_mrae


def test_mnist_cifar10(model, train_name, dataset, loss_mrae, cuda_device):
    seed = 2032
    print("Testing the model...")
    n_classes = 10
    n_samples_test = 5000
    true_prevs = pd.DataFrame(columns=np.arange(n_classes), index=range(n_samples_test), dtype="float")
    results = pd.DataFrame(columns=np.arange(n_classes), index=range(n_samples_test), dtype="float")
    results_errors = pd.DataFrame(columns=("AE", "MSE", "RAE"), index=range(n_samples_test), dtype="float")
    if dataset == "mnist":
        testset = load_mnist(train=False, data_augmentation=False, seed=seed)
    elif dataset == "fashionmnist":
        testset = load_fashionmnist(train=False, data_augmentation=False, seed=seed)
    elif dataset == "cifar10":
        testset = load_cifar10(train=False, data_augmentation=False, seed=seed)

    test_bag_generator = APPBagGenerator(device=cuda_device, seed=seed)
    if isinstance(testset, TensorDataset):
        bags, prevalences = test_bag_generator.compute_bags(n_samples_test, 500, testset.tensors[1])
    else:
        bags, prevalences = test_bag_generator.compute_bags(n_samples_test, 500, testset.targets)

    for i in tqdm(range(n_samples_test)):
        sample = bags[i]
        dataset = torch.utils.data.Subset(testset, sample)
        p_hat = model.predict(dataset)
        results.iloc[i] = p_hat
        p_hat = torch.FloatTensor(p_hat).to(cuda_device)
        results_errors.iloc[i]["AE"] = torch.nn.functional.l1_loss(p_hat, prevalences[i, :]).cpu().numpy()
        results_errors.iloc[i]["MSE"] = torch.nn.functional.mse_loss(p_hat, prevalences[i, :]).cpu().numpy()
        results_errors.iloc[i]["RAE"] = loss_mrae(prevalences[i, :], p_hat).cpu().numpy()

    true_prevs[:] = prevalences.cpu().numpy()
    results.to_csv(os.path.join("results/", train_name + ".txt"), index_label="id")
    results_errors.to_csv(os.path.join("results/", train_name + "_errors.txt"), index_label="id")
    true_prevs.to_csv(os.path.join("results/", train_name + "_true.txt"), index_label="id")
    print(results_errors.describe())


if __name__ == "__main__":
    # Parametrice the script with argparse
    parser = argparse.ArgumentParser(description="MNIST/FASHIONMNIST/CIFAR training script")
    parser.add_argument("-t", "--train_name", help="Name for this training", required=True)
    parser.add_argument("-n", "--network", help="network to use: histnet, settransformers, deepsets", required=True)
    parser.add_argument("-p", "--network_parameters", help="File with the specific network parameters")
    parser.add_argument("-d", "--dataset", help="Dataset to use: MNIST, FASHIONMNIST, CIFAR10", required=True)
    parser.add_argument("-w", "--load_weights", help="Loads network weight for the FE part", action="store_true")
    parser.add_argument("-l", "--loss_func", help="Loss function (ae, rae, mse)", required=True)
    parser.add_argument("-c", "--cuda_device", help="Device cuda:0 or cuda:1", required=True)
    print("Using following arguments:")
    args = vars(parser.parse_args())
    print(args)

    args["cuda_device"] = torch.device(args["cuda_device"])

    model, loss_mrae = train_mnist_cifar10(**args)
    test_mnist_cifar10(model, args["train_name"], args["dataset"], loss_mrae, args["cuda_device"])
