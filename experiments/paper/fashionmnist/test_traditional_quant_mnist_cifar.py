from quantificationlib.qbase import AC, PAC
from quantificationlib.qbase import CC, PCC
from quantificationlib.qbase import EM
import argparse
import numpy as np
from tqdm import tqdm
from dlquantification.utils.utils import APPBagGenerator
from dlquantification.utils.lossfunc import MRAE
import pandas as pd
import torch
import os
from abstention.calibration import TempScaling

# from abstention.label_shift import EMImbalanceAdapter, BBSEImbalanceAdapter


if __name__ == "__main__":
    # Parametrice the script with argparse
    parser = argparse.ArgumentParser(description="MNIST/CIFAR finetuning script")
    parser.add_argument("-d", "--dataset", help="Dataset to use: MNIST, FASHIONMNIST, CIFAR10", required=True)
    print("Using following arguments:")
    args = vars(parser.parse_args())
    print(args)
    dataset = args["dataset"]

    method_name = [
        "CC",
        "AC",
        "PCC",
        "PAC",
        "EM",
        "EM-BCTS"
    ]

    cc = CC()
    ac = AC()
    pcc = PCC()
    pac = PAC()
    em = EM()
    # em = EMImbalanceAdapter()
    # em_nbvs = EMImbalanceAdapter(calibrator_factory=NoBiasVectorScaling())
    # em_bcts = EMImbalanceAdapter(calibrator_factory=TempScaling())
    # em_vs = EMImbalanceAdapter(calibrator_factory=VectorScaling())
    # bbse_hard = BBSEImbalanceAdapter(soft=False)
    # bbse_soft = BBSEImbalanceAdapter(soft=True)

    predictions_val = np.loadtxt("predictions/output_predictions_{}_val.csv".format(dataset))
    y_val = np.loadtxt("predictions/true_{}_val.csv".format(dataset), dtype="int")
    predictions_test = np.loadtxt("predictions/output_predictions_{}_test.csv".format(dataset))
    y_test = np.loadtxt("predictions/true_{}_test.csv".format(dataset), dtype="int")

    # CALIBRATION METHODS FOR EM
    bcts = TempScaling(bias_positions="all")
    bcts = bcts(predictions_val, np.eye(10)[y_val], posterior_supplied=True)
    # --------------------------

    cc.fit(X=None, y=y_val, predictions_train=predictions_val)
    ac.fit(X=None, y=y_val, predictions_train=predictions_val)
    pcc.fit(X=None, y=y_val, predictions_train=predictions_val)
    pac.fit(X=None, y=y_val, predictions_train=predictions_val)
    em.fit(X=None, y=y_val, predictions_train=predictions_val)

    n_samples_test = 5000
    torch.manual_seed(2032)

    device = torch.device("cpu")

    test_bag_generator = APPBagGenerator(device, seed=2032)
    bags, prevalences = test_bag_generator.compute_bags(n_samples_test, 500, y_test)

    results_mae = pd.DataFrame(np.zeros((n_samples_test, len(method_name))), columns=method_name)
    results_rmae = pd.DataFrame(np.zeros((n_samples_test, len(method_name))), columns=method_name)
    results_mse = pd.DataFrame(np.zeros((n_samples_test, len(method_name))), columns=method_name)

    estim_prevs = {}
    for m_name in method_name:
        estim_prevs[m_name] = pd.DataFrame(columns=np.arange(10), index=range(n_samples_test), dtype="float")

    loss_mrae = MRAE(eps=1.0 / (2 * 500), n_classes=10).MRAE

    for i in tqdm(range(n_samples_test)):
        sample = bags[i]
        predictions_sample = predictions_test[sample.cpu().numpy(), :]
        p_hats = [
            cc.predict(X=None, predictions_test=predictions_sample),
            ac.predict(X=None, predictions_test=predictions_sample),
            pcc.predict(X=None, predictions_test=predictions_sample),
            pac.predict(X=None, predictions_test=predictions_sample),
            em.predict(X=None, predictions_test=predictions_sample),
            em.predict(X=None, predictions_test=bcts(predictions_sample)),
        ]

        for n_method, p_hat in enumerate(p_hats):
            estim_prevs[method_name[n_method]].iloc[i, :] = p_hat
            p_hat = torch.FloatTensor(p_hat).to(device)
            results_mae.iloc[i, n_method] = torch.nn.functional.l1_loss(p_hat, prevalences[i, :]).cpu().numpy()
            results_rmae.iloc[i, n_method] = loss_mrae(prevalences[i, :], p_hat).cpu().numpy()
            results_mse.iloc[i, n_method] = torch.nn.functional.mse_loss(prevalences[i, :], p_hat).cpu().numpy()

    print(results_mae.describe())
    print(results_rmae.describe())
    print(results_mse.describe())

    for method in method_name:
        true_prevs = pd.DataFrame(columns=np.arange(10), index=range(n_samples_test), dtype="float")
        results_errors = pd.DataFrame(columns=("AE", "RAE", "MSE"), index=range(n_samples_test), dtype="float")
        true_prevs[:] = prevalences.cpu().numpy()
        results_errors["AE"] = results_mae.loc[:, method]
        results_errors["RAE"] = results_rmae.loc[:, method]
        results_errors["MSE"] = results_mse.loc[:, method]
        true_prevs.to_csv(os.path.join("results/", "{}_{}_true.txt".format(method, dataset)), index_label="id")
        results_errors.to_csv(os.path.join("results/", "{}_{}_errors.txt".format(method, dataset)), index_label="id")
        estim_prevs[method].to_csv(os.path.join("results/", "{}_{}.txt".format(method, dataset)), index_label="id")
