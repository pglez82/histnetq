import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from dlquantification.transnet import TransNet
from dlquantification.utils.utils import UnlabeledBagGenerator, UnlabeledMixerBagGenerator
from dlquantification.featureextraction.fullyconnected import FCFeatureExtractionModule
import os
from dlquantification.utils.lossfunc import MRAE
import json
import optuna
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
import sys


def train_lequa(settransformer_parameters, epoch_callback, cuda_device):
    n_features = 300
    path = "./lequa/T1B/public"
    common_param_path = "parameters/common_parameters_T1B.json"
    n_train_samples = 700
    n_val_samples = 300
    n_samples = 1000
    sample_size = 1000
    fe_hidden_sizes = [1024]
    fe_output_size = 512
    real_bags_proportion = 0.5

    print("Loading dataset T1B ...", end="")
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

    x_unlabeled_train = torch.from_numpy(x_unlabeled_train).to(cuda_device)
    x_unlabeled_val = torch.from_numpy(x_unlabeled_val).to(cuda_device)
    dataset_train = TensorDataset(x_unlabeled_train)
    dataset_val = TensorDataset(x_unlabeled_val)
    print("Done.")
    # ---------------Load unlabeled data----------------------
    with open(common_param_path, "r") as f:
        common_parameters = json.loads(f.read())

    seed = 2032

    # Bag generators
    train_bag_generator = UnlabeledMixerBagGenerator(
        cuda_device,
        prevalences=train_prevalences,
        sample_size=sample_size,
        real_bags_proportion=real_bags_proportion,
        seed=seed,
    )
    val_bag_generator = UnlabeledBagGenerator(cuda_device, val_prevalences, sample_size, pick_all=True, seed=seed)

    # Loss function
    n_classes = 28
    loss_mrae = MRAE(eps=1.0 / (2 * sample_size), n_classes=n_classes).MRAE

    torch.manual_seed(seed)

    # ADAMW works well sith 0.0001 and 0.000001 (lr)
    fe = FCFeatureExtractionModule(
        input_size=300, output_size=fe_output_size, hidden_sizes=fe_hidden_sizes, dropout=0.5
    )

    parameters = {**common_parameters, **settransformer_parameters}
    parameters["n_classes"] = n_classes
    parameters["random_seed"] = seed
    parameters["feature_extraction_module"] = fe
    parameters["bag_generator"] = train_bag_generator
    parameters["val_bag_generator"] = val_bag_generator
    parameters["device"] = cuda_device
    parameters["quant_loss"] = loss_mrae
    parameters["dataset_name"] = "T1B"
    parameters["use_wandb"] = False
    parameters["callback_epoch"] = epoch_callback
    print("Network parameteres: ", parameters)

    model = TransNet(**parameters)
    best_loss = model.fit(dataset=dataset_train, val_dataset=dataset_val)

    return best_loss


def objective(trial):
    def epoch_callback(loss, epoch):
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    transformer_parameters = {}
    transformer_parameters["trans_out_size"] = trial.suggest_int("trans_out_size", 32, 2048)
    transformer_parameters["trans_ind_points"] = trial.suggest_int("trans_ind_points", 2, 500)
    transformer_parameters["trans_hidden_dim"] = trial.suggest_int("trans_hidden_dim", 32, 2048, step=32)
    # Number of heads should be multiple of hidden_dim
    transformer_parameters["trans_heads"] = trial.suggest_categorical("trans_heads", [2, 4, 8, 16, 32])
    return train_lequa(transformer_parameters, epoch_callback, device)


if __name__ == "__main__":
    device = torch.device(sys.argv[1])

    pruner = HyperbandPruner()
    study = optuna.create_study(
        direction="minimize",
        study_name="T1B_transformers",
        storage="sqlite:///T1B_optuna_transformers.db",
        load_if_exists=True,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=10000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
