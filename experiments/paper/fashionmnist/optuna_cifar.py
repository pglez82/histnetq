import torch
from dlquantification.histnet import HistNet
from dlquantification.featureextraction.cnn import CNNFeatureExtractionModuleCifar10
import json
import optuna
from utils import load_cifar10
from dlquantification.utils.utils import APPBagGenerator
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
import sys


def train_cifar10(parameters_optimize, epoch_callback, cuda_device):
    dataset_train, dataset_val = load_cifar10(train=True, data_augmentation=False)
    common_param_path = "../parameters/common_parameters_cifar10.json"

    with open(common_param_path, "r") as f:
        common_parameters = json.loads(f.read())

    seed = 2032
    n_classes = 10

    # Bag generators
    train_bag_generator = APPBagGenerator(device=cuda_device, seed=seed)
    val_bag_generator = APPBagGenerator(device=cuda_device, seed=seed)

    fe = CNNFeatureExtractionModuleCifar10(output_size=parameters_optimize["output_size"])
    del parameters_optimize["output_size"]

    parameters = {**common_parameters}
    parameters["n_classes"] = n_classes
    parameters["histogram"] = "hard"
    parameters["random_seed"] = seed
    parameters["feature_extraction_module"] = fe
    parameters["bag_generator"] = train_bag_generator
    parameters["val_bag_generator"] = val_bag_generator
    parameters["device"] = cuda_device
    parameters["quant_loss"] = torch.nn.L1Loss()
    parameters["use_wandb"] = False
    parameters["patience"] = 10
    parameters["callback_epoch"] = epoch_callback

    for key, value in parameters_optimize.items():
        parameters[key] = value

    print("Network parameteres: ", parameters)

    model = HistNet(**parameters)
    best_loss = model.fit(dataset=dataset_train, val_dataset=dataset_val)

    return best_loss


def objective(trial):
    def epoch_callback(loss, epoch):
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    parameters_optimize = {}
    # list of parameters to optimize
    parameters_optimize["output_size"] = trial.suggest_int("output_size", 16, 1024, step=16)
    parameters_optimize["n_bins"] = trial.suggest_int("n_bins", 4, 64)
    parameters_optimize["dropout"] = trial.suggest_float("dropout_quant", 0, 0.8)
    parameters_optimize["gradient_accumulation"] = trial.suggest_int("gradient_accumulation", 1, 250)
    num_linear_layers = trial.suggest_int("num_linear_layers", 1, 3)
    parameters_optimize["linear_sizes"] = []
    for i in range(num_linear_layers):
        parameters_optimize["linear_sizes"].append(trial.suggest_int("linear_sizes{}".format(i), 4, 2048, step=4))
    parameters_optimize["start_lr"] = trial.suggest_float("lr", 0.00001, 0.01, log=True)
    parameters_optimize["weight_decay"] = trial.suggest_float("weight_decay", 0.000001, 0.1, log=True)
    return train_cifar10(parameters_optimize, epoch_callback, device)


if __name__ == "__main__":
    device = torch.device(sys.argv[1])

    pruner = HyperbandPruner()
    study = optuna.create_study(
        direction="minimize",
        study_name="CIFAR10",
        storage="sqlite:///cifar10_optuna.db",
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
