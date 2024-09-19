"""HistNet implementation. It contains actual HistNet code."""

import torch
from dlquantification.quantmodule.histograms.SigmoidHistogram import SigmoidHistogram
from dlquantification.quantmodule.histograms.SoftHistogram import SoftHistogram, SoftHistogramRBF
from dlquantification.quantmodule.histograms.HardHistogram import HardHistogram
from dlquantification.quantmodule.histograms.SigmoidHistogramBatched import SigmoidHistogramBatched
from dlquantification.quantmodule.histograms.SoftHistogramBatched import SoftHistogramBatched, SoftHistogramRBFBatched
from dlquantification.quantmodule.histograms.HardHistogramBatched import HardHistogramBatched
from dlquantification.utils.utils import BaseBagGenerator
import torch.nn.functional as F

from dlquantification.dlquantification import DLQuantification


class HistNetHistograms(torch.nn.Module):

    """HistNet Pytorch module.

    This is the full network definition: feature extraction, histogram, and fully connected layers to do the
    quantification.

    :param n_classes: Number of classes to use in the problem
    :type n_classes: int
    :param histogram: Histogram layer. Check histograms module to see options available
    :type histogram: class
    :param dropout: Dropout to use in the last linear layers, after the histogram
    :type dropout: float
    :param feature_extraction_module: Feature extraction part of the network. Check feature extration module to see
                                      options available
    :type feature_extraction_module: class
    :param linear_sizes: List of ints with the sizes of the linear layers after the  histogram. Last layer is implicit
                                      and we do not need to specify it
    :type linear_sizes: list
    :param use_labels: Extra connection for the labels
    :type use_labels: bool
    :param use_cnn: Uses an extra layer of type CNN after the histogram
    :type use_labels: bool
    """

    def __init__(self, histogram, input_size, n_bins=8, quantiles=False):

        super(HistNetHistograms, self).__init__()
        self.output_size = n_bins * input_size
        # Create the quantification module for HistNet
        if histogram == "sigmoid":
            # TODO: We need to check the sigma value. Initial value was 1000 but it is ok 10000?.
            self.histogram = SigmoidHistogram(num_bins=n_bins, min=0, max=1, sigma=1000, quantiles=quantiles)
        elif histogram == "sigmoid_batched":
            self.histogram = SigmoidHistogramBatched(num_bins=n_bins, min=0, max=1, sigma=1000, quantiles=quantiles)
        elif histogram == "softrbf":
            self.histogram = SoftHistogramRBF(
                n_features=input_size,
                num_bins=n_bins,
                quantiles=quantiles,
            )
        elif histogram == "softrbf_batched":
            self.histogram = SoftHistogramRBFBatched(
                n_features=input_size,
                num_bins=n_bins,
                quantiles=quantiles,
            )
        elif histogram == "hard":
            self.histogram = HardHistogram(
                n_features=input_size,
                num_bins=n_bins,
                quantiles=quantiles,
            )
        elif histogram == "hard_batched":
            self.histogram = HardHistogramBatched(
                n_features=input_size,
                num_bins=n_bins,
                quantiles=quantiles,
            )
        elif histogram == "soft":
            self.histogram = SoftHistogram(
                n_features=input_size,
                num_bins=n_bins,
                quantiles=quantiles,
            )
        elif histogram == "soft_batched":
            self.histogram = SoftHistogramBatched(
                n_features=input_size,
                num_bins=n_bins,
                quantiles=quantiles,
            )
        else:
            raise ValueError("Invalid histogram type, expected one of [sigmoid,soft,softrbf,hard]")

        self.layers = torch.nn.Sequential()
        self.layers.add_module("sigmoid", torch.nn.Sigmoid())
        self.layers.add_module("histogram", self.histogram)

    def forward(self, input):
        return self.layers(input)


class HistNet(DLQuantification):
    """
    Class for using the HistNet quantifier.

    HistNet builds creates artificial samples with fixed size and learns from them. Every example in each sample goes
    through the network and we build a histogram with all the examples in a sample. This is used in the quantification
    module where we use this vector to quantify the sample.

    :param train_epochs: How many times to repeat the process of going over training data. Each epoch will train over
                         n_bags samples.
    :type train_epochs: int
    :param test_epochs: How many times to repeat the process over the testing data (returned prevalences are averaged).
    :type test_epochs: int
    :param start_lr: Learning rate for the network (initial value).
    :type start_lr: float
    :param end_lr: Learning rate for the network. The value will be decreasing after a few epochs without improving
                   (check patiente parameter).
    :type end_lr: float
    :param n_classes: Number of classes
    :type n_classes: int
    :param optimizer_class: torch.optim class to make the optimization. Example torch.optim.Adam
    :type optimizer_class: class
    :param lr_factor: Learning rate decrease factor after patience epochs have passed without improvement.
    :type lr_factor: float
    :param batch_size: Update weights after this number of samples.
    :type batch_size: int
    :param patience: Number of epochs after which we will decrease the learning rate if there is no improvement.
    :type patience: int
    :param n_bags: How many artificial samples to build per epoch. If we get a single value this is used for training,
                   val and test. If a tuple with three values is provided it will used as (n_bags_train,n_bags_val,
                   n_bags_test)
    :type n_bags: int or (int,int,int)
    :param bag_size: Number of examples per sample (train,val,test).
    :type bag_size: int or (int,int,int)
    :param bag_generator: Class that will be in charge of generating the samples.
    :type bag_generator: class
    :param val_bag_generator: Class that will be in charge of generating the validation samples.
    :type val_bag_generator: class
    :param test_bag_generator: Class that will be in charge of generating the test samples.
    :type test_bag_generator: class
    :param n_bins: Number of bins used to build the histogram.
    :type n_bins: int
    :param random_seed: Seed to make results reproducible. This net needs to generate the bags so the seed is important.
    :type random_seed: int
    :param dropout: Dropout to use in the network (avoid overfitting).
    :type dropout: float
    :param weight_decay: L2 regularization for the model.
    :type weight_decay: float
    :param val_split: By default we validate using the train data. If a split is given, we partition the data for using
                      it as validation and early stopping. We can receive the split in different ways: 1) float:
                      percentage of data reserved for validation. 2) int: if 0, training set is used as validation.
                      If any other number, this number of examples will be used for validation. 3) tuple: if we get a
                      tuple, this will be the specific indexes used for validation
    :type val_split: int, float or tuple
    :param quant_loss: loss function to optimize in the quantification problem. Classification loss if use_labels=True
                       is fixed (CrossEntropyLoss used)
    :type quant_loss: function
    :param epsilon: If the error is less than this number, do not update the weights in this iteration.
    :type epsilon: float
    :param feature_extraction_module: Pytorch module with the feature extraction layers.
    :type feature_extraction_module: torch.Module
    :param linear_sizes: Tuple or list with the sizes of the linear layers used in the quantification module.
    :type linear_sizes: tuple
    :param histogram: Which histogram to use (sigmoid, soft, softrbf, hard)
    :type histogram: str
    :param quantiles: If true, use a quantile version of the histogram.
    :type quantiles: boolean
    :param use_labels: If true, use the class labels to help fit the feature extraction module of the network. A mix of
                       quant_loss + CrossEntropyLoss will be used as the loss in this case.
    :type use_labels: boolean
    :param use_labels_epochs: After this number of epochs, do not use the labels anymore. By default is use_labels is
                              true, labels are going to be used for all the epochs.
    :type use_labels_epochs: int
    :param output_function: Output function to use. Possible values 'softmax' or 'normalize'. Both will end up with a
                            probability distribution adding one
    :type output_function: str
    :param use_cnn: If true, add an extra cnn layer after the histogram
    :type use_cnn: boolean
    :param num_workers: Number of workers to use in the dataloaders. Note that if you choose to use more than one worker
                        you will need to use device=torch.device('cpu') in the bag generators, if not, an exception
                        will be raised.
    :type num_workers: int
    :param use_fp16: If true, trains using half precision mode.
    :type use_fp16: boolean
    :param device: Device to use for training/testing.
    :type device: torch.device
    :param callback_epoch: Function to call after each epoch. Useful to optimize with Optuna
    :type callback_epoch: function
    :param save_model_path: File to save the model when trained. We also load it if exists to skip training.
    :type save_model_path: file
    :param save_checkpoint_epochs: Save a checkpoint every n epochs. This parameter needs save_model_path to be set as
    it reuses the name of the file but appending the extension ckpt to it.
    :type save_checkpoint_epochs: int
    :param tensorboard_dir: Path to a directory where to store tensorboard logs. We can explore them using
                            tensorboard --logdir directory. By default no logs are saved.
    :type tensorboard_dir: str
    :param use_wandb: If true, we use wandb to log the training.
    :type use_wandb: bool
    :param wandb_experiment_name: Name of the experiment in wandb.
    :type wandb_experiment_name: str
    :param log_samples: If true the network will log all the generated samples with p and p_hat and the loss (for
                        training and validation)
    :type log_samples:  bool
    :param verbose: Verbose level.
    :type verbose: int
    :param dataset_name: Only for logging purposes.
    :type dataset_name: str
    """

    def __init__(
        self,
        train_epochs,
        test_epochs,
        n_classes,
        start_lr,
        end_lr,
        n_bags,
        bag_size,
        n_bins: int,
        random_seed,
        linear_sizes,
        feature_extraction_module,
        batch_size: int,
        bag_generator: BaseBagGenerator,
        gradient_accumulation: int = None,
        val_bag_generator: BaseBagGenerator = None,
        test_bag_generator: BaseBagGenerator = None,
        optimizer_class=torch.optim.AdamW,
        dropout: float = 0,
        weight_decay: float = 0,
        lr_factor=0.1,
        val_split=0,
        quant_loss=torch.nn.L1Loss(),
        epsilon=0,
        histogram="sigmoid",
        quantiles: bool = False,
        output_function="softmax",
        use_labels: bool = False,
        use_labels_epochs=None,
        batch_size_fe=None,
        device=torch.device("cpu"),
        patience: int = 20,
        num_workers: int = 0,
        use_fp16: bool = False,
        callback_epoch=None,
        save_model_path=None,
        save_checkpoint_epochs=None,
        verbose=0,
        tensorboard_dir=None,
        use_wandb: bool = False,
        wandb_experiment_name: str = None,
        log_samples=False,
        dataset_name="",
    ):
        torch.manual_seed(random_seed)

        # Init the model
        quantmodule = HistNetHistograms(
            histogram=histogram, input_size=feature_extraction_module.output_size, n_bins=n_bins, quantiles=quantiles
        )

        super().__init__(
            train_epochs=train_epochs,
            test_epochs=test_epochs,
            n_classes=n_classes,
            start_lr=start_lr,
            end_lr=end_lr,
            n_bags=n_bags,
            bag_size=bag_size,
            random_seed=random_seed,
            batch_size=batch_size,
            quantmodule=quantmodule,
            bag_generator=bag_generator,
            val_bag_generator=val_bag_generator,
            test_bag_generator=test_bag_generator,
            optimizer_class=optimizer_class,
            weight_decay=weight_decay,
            lr_factor=lr_factor,
            val_split=val_split,
            quant_loss=quant_loss,
            batch_size_fe=batch_size_fe,
            gradient_accumulation=gradient_accumulation,
            feature_extraction_module=feature_extraction_module,
            linear_sizes=linear_sizes,
            dropout=dropout,
            epsilon=epsilon,
            output_function=output_function,
            use_labels=use_labels,
            use_labels_epochs=use_labels_epochs,
            device=device,
            patience=patience,
            num_workers=num_workers,
            use_fp16=use_fp16,
            callback_epoch=callback_epoch,
            save_model_path=save_model_path,
            save_checkpoint_epochs=save_checkpoint_epochs,
            verbose=verbose,
            tensorboard_dir=tensorboard_dir,
            use_wandb=use_wandb,
            wandb_experiment_name=wandb_experiment_name,
            log_samples=log_samples,
            dataset_name=dataset_name,
        )
