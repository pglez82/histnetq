import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from dlquantification.transnet import TransNet
from dlquantification.histnet import HistNet
from dlquantification.deepsets import DeepSets
from dlquantification.utils.utils import UnlabeledBagGenerator, UnlabeledMixerBagGenerator
from dlquantification.featureextraction.fullyconnected import FCFeatureExtractionModule
from dlquantification.featureextraction.nofe import NoFeatureExtractionModule
from dlquantification.featureextraction.transfomers import ISABExtractionModule
import os
from dlquantification.utils.lossfunc import MRAE
import json
import argparse
from tqdm import tqdm


def get_n_classes(dataset):
    if dataset == "T1A" or dataset == "T1":
        return 2
    elif dataset == "T1B" or dataset == "T2":
        return 28
    else:
        raise ValueError("Dataset is not correct")


def train_lequa(train_name, network, network_parameters, dataset, feature_extraction="rff", skip_sample_mixer=False, cuda_device="cuda:0"):
    n_features = 256 if (dataset=='T2' or dataset=='T1') else 300

    if dataset == "T1A":
        path = "lequa/T1A/public"
        common_param_path = "parameters/common_parameters_T1A.json"
        n_train_samples = 700
        n_val_samples = 300
        n_samples = 1000
        sample_size = 250
        fe_hidden_sizes = [1024, 1024]
        fe_output_size = 300
        real_bags_proportion = 0.1
    elif dataset == "T1B":
        path = "lequa/T1B/public"
        common_param_path = "parameters/common_parameters_T1B.json"
        n_train_samples = 700
        n_val_samples = 300
        n_samples = 1000
        sample_size = 1000
        fe_hidden_sizes = [1024]
        fe_output_size = 512
        real_bags_proportion = 0.5
    if dataset == "T1":
        path = "lequa/T1/public"
        common_param_path = "parameters/common_parameters_T1.json"
        n_train_samples = 700
        n_val_samples = 300
        n_samples = 1000
        sample_size = 250
        fe_hidden_sizes = [1024, 1024]
        fe_output_size = 512
        real_bags_proportion = 0.1
    elif dataset == "T2":
        path = "lequa/T2/public"
        common_param_path = "parameters/common_parameters_T2.json"
        n_train_samples = 700
        n_val_samples = 300
        n_samples = 1000
        sample_size = 1000
        fe_hidden_sizes = [1024]
        fe_output_size = 512
        real_bags_proportion = 0.5


    if skip_sample_mixer:
        print("Warning: skipping sample mixer (for ablation study)")
        real_bags_proportion=1
        
    print("Loading dataset %s ... " % dataset, end="")
    x_unlabeled_train = np.zeros((n_train_samples * sample_size, n_features)).astype(np.float32)
    x_unlabeled_val = np.zeros((n_val_samples * sample_size, n_features)).astype(np.float32)
    prevalences = pd.read_csv(os.path.join(path, "dev_prevalences.txt"))
    train_prevalences = torch.from_numpy(prevalences.iloc[0:n_train_samples, 1:].to_numpy().astype(np.float32)).to(
        cuda_device
    )
    val_prevalences = torch.from_numpy(
        prevalences.iloc[n_train_samples : n_train_samples + n_val_samples, 1:].to_numpy().astype(np.float32)
    ).to(cuda_device)

    for i in range(n_samples):
        sample = pd.read_csv(os.path.join(path, "dev_samples/{}.txt".format(i)))
        if i < n_train_samples:
            x_unlabeled_train[
                i * sample_size : (i + 1) * sample_size,
            ] = sample.to_numpy()
        else:
            j = i - n_train_samples
            x_unlabeled_val[
                j * sample_size : (j + 1) * sample_size,
            ] = sample.to_numpy()

    x_unlabeled_train = torch.from_numpy(x_unlabeled_train)
    x_unlabeled_val = torch.from_numpy(x_unlabeled_val)

    if dataset == "T2" or dataset == "T1":
        mean = x_unlabeled_train.mean(dim=0)
        std = x_unlabeled_train.std(dim=0)

        torch.save({'mean': mean, 'std': std}, 'mean_std_{}.pth'.format(dataset))

        x_unlabeled_train = (x_unlabeled_train - mean) / std
        x_unlabeled_val = (x_unlabeled_val - mean) / std

    dataset_train = TensorDataset(x_unlabeled_train)
    dataset_val = TensorDataset(x_unlabeled_val)
    print("Done.")
    # ---------------Load unlabeled data----------------------
    with open(common_param_path, "r") as f:
        common_parameters = json.loads(f.read())
    with open(network_parameters, "r") as f:
        network_parameters = json.loads(f.read())

    seed = 2032

    # Bag generators
    train_bag_generator = UnlabeledMixerBagGenerator(
        'cpu',
        prevalences=train_prevalences,
        sample_size=sample_size,
        real_bags_proportion=real_bags_proportion,
        seed=seed,
    )
    val_bag_generator = UnlabeledBagGenerator('cpu', val_prevalences, sample_size, pick_all=True, seed=seed)

    # Loss function
    n_classes = get_n_classes(dataset)
    loss_mrae = MRAE(eps=1.0 / (2 * sample_size), n_classes=n_classes).MRAE

    torch.manual_seed(seed)

    if feature_extraction == "rff":
        fe = FCFeatureExtractionModule(
            input_size=n_features, output_size=fe_output_size, hidden_sizes=fe_hidden_sizes, dropout=0.5
        )
    elif feature_extraction == "nofe":
        fe = NoFeatureExtractionModule(input_size=n_features)
    elif feature_extraction == "ISAB":
        fe = ISABExtractionModule(input_size=n_features, dim_hidden=256, num_heads=4, num_inds=128, ln=False)
    elif feature_extraction == "ISAB_rFF":
        rFF = FCFeatureExtractionModule(
            input_size=256, output_size=fe_output_size, hidden_sizes=fe_hidden_sizes, dropout=0.5
        )
        fe = ISABExtractionModule(input_size=n_features, dim_hidden=256, num_heads=4, num_inds=16, ln=False, rFF=rFF)

    parameters = {**common_parameters, **network_parameters}
    parameters["n_classes"] = n_classes
    parameters["random_seed"] = seed
    parameters["feature_extraction_module"] = fe
    parameters["bag_generator"] = train_bag_generator
    parameters["val_bag_generator"] = val_bag_generator
    parameters["device"] = cuda_device
    parameters["quant_loss"] = loss_mrae
    parameters["dataset_name"] = dataset
    parameters["tensorboard_dir"] = "runs/" + dataset
    parameters["save_model_path"] = "savedmodels/" + train_name + ".pkl"
    parameters["wandb_experiment_name"] = train_name
    parameters["use_wandb"] = True
    print("Network parameteres: ", parameters)

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


def test_lequa(model, train_name, dataset, loss_mrae, cuda_device):
    print("Testing the model...")
    
    if dataset == "T2" or dataset == "T1":
        meanstd = torch.load('mean_std_{}.pth'.format(dataset))
        mean = meanstd['mean']
        std = meanstd['std']
    
    n_classes = get_n_classes(dataset)
    samples_to_predict_path = "lequa/" + dataset + "/public/test_samples/"
    prevalences = pd.read_csv(os.path.join("lequa/" + dataset + "/public/test_prevalences.txt"))
    results = pd.DataFrame(columns=np.arange(n_classes), index=range(5000), dtype="float")
    results_errors = pd.DataFrame(columns=("AE", "RAE"), index=range(5000), dtype="float")
    for i in tqdm(range(5000)):
        sample = pd.read_csv(os.path.join(samples_to_predict_path, "{}.txt".format(i)))
        sample = torch.from_numpy(sample.to_numpy().astype(np.float32))
        if dataset == "T2" or dataset == "T1":
            sample = (sample - mean) / std
        sample = TensorDataset(sample)
        p_hat = model.predict(sample)
        results.iloc[i] = p_hat
        results_errors.iloc[i]["AE"] = torch.nn.functional.l1_loss(
            torch.FloatTensor(p_hat), torch.FloatTensor(prevalences.iloc[i, 1:])
        ).numpy()
        results_errors.iloc[i]["RAE"] = loss_mrae(
            torch.FloatTensor(prevalences.iloc[i, 1:]), torch.FloatTensor(p_hat)
        ).numpy()

    results.to_csv(os.path.join("results/", train_name + ".txt"), index_label="id")
    results_errors.to_csv(os.path.join("results/", train_name + "_errors.txt"), index_label="id")
    print(results_errors.describe())


if __name__ == "__main__":
    # Parametrice the script with argparse
    parser = argparse.ArgumentParser(description="LEQUA training script")
    parser.add_argument("-t", "--train_name", help="Name for this training", required=True)
    parser.add_argument("-n", "--network", help="network to use: histnet, settransformers, deepsets", required=True)
    parser.add_argument("-p", "--network_parameters", help="File with the specific network parameters")
    parser.add_argument("-f", "--feature_extraction", help="nofe, rff, isab")
    parser.add_argument("-m", "--skip_sample_mixer",action="store_true", help="Add this parameter to skip the sample mixer")
    parser.add_argument("-d", "--dataset", help="Dataset to use: lequaT1A, lequaT1B", required=True)
    parser.add_argument("-c", "--cuda_device", help="Device cuda:0 or cuda:1", required=True)
    print("Using following arguments:")
    args = vars(parser.parse_args())
    print(args)

    args["cuda_device"] = torch.device(args["cuda_device"])

    model, loss_mrae = train_lequa(**args)
    test_lequa(model, args["train_name"], args["dataset"], loss_mrae, args["cuda_device"])
