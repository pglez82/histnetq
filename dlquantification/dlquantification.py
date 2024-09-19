"""HistNet implementation. It contains actual HistNet code."""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data.dataset import TensorDataset
import copy
import time
from dlquantification.utils.utils import BagSampler, BaseBagGenerator, batch_collate_fn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import os.path
from datetime import datetime
from functools import partial
import wandb


class View(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = (shape,)  # extra comma

    def forward(self, x):
        return x.view(*self.shape)


class Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, p=1, dim=1)


class DLQuantificationModule(torch.nn.Module):
    def __init__(
        self,
        n_classes,
        dropout,
        feature_extraction_module,
        quantmodule,
        linear_sizes,
        use_labels,
        output_function,
        residual_connection,
        batch_size_fe=None,
    ):
        super(DLQuantificationModule, self).__init__()
        self.feature_extraction_module = feature_extraction_module
        self.quantmodule = quantmodule
        self.use_labels = use_labels
        self.residual_connection = residual_connection
        self.batch_size_fe = batch_size_fe
        if use_labels:
            self.classification_module = torch.nn.Sequential(
                torch.nn.Linear(feature_extraction_module.output_size, n_classes)
            )
        if residual_connection:
            self.residual_connection = torch.nn.Linear(feature_extraction_module.output_size, quantmodule.output_size)

        self.output_module = torch.nn.Sequential()
        prev_size = quantmodule.output_size

        for i, linear_size in enumerate(linear_sizes):
            extra_linear = torch.nn.Linear(prev_size, linear_size)
            self.output_module.add_module("quant_linear_%d" % i, extra_linear)
            # torch.nn.init.eye_(extra_linear.weight)
            # torch.nn.init.constant_(extra_linear.bias, 0)
            self.output_module.add_module("quant_leaky_relu_%d" % i, torch.nn.LeakyReLU())
            self.output_module.add_module("quant_dropout_%d" % i, torch.nn.Dropout(dropout))
            prev_size = linear_size

        # Last layer with size equal to the number of classes
        last_linear = torch.nn.Linear(prev_size, n_classes)
        if prev_size == n_classes:
            torch.nn.init.eye_(last_linear.weight)
            torch.nn.init.constant_(last_linear.bias, 0)
        self.output_module.add_module("quant_last_linear", last_linear)
        if output_function == "softmax":
            self.output_module.add_module("quant_softmax", torch.nn.Softmax(dim=1))
        else:
            self.output_module.add_module("quant_norm_relu", torch.nn.ReLU())
            self.output_module.add_module("quant_norm", Normalize())

    def forward(self, input, return_classification=False):
        if self.batch_size_fe is not None:
            features = torch.empty(
                self.__getinputsize(input),
                self.feature_extraction_module.output_size,
                dtype=torch.float32,
                device=self.__getinputdevice(input),
            )
            for i, minibatch in enumerate(self.__create_minibatches_input(input, self.batch_size_fe)):
                features[
                    i * self.batch_size_fe : (i + 1) * self.batch_size_fe,
                ] = self.feature_extraction_module(minibatch)
        else:
            features = self.feature_extraction_module(input)

        if self.residual_connection:
            self.residual_connection(features)

        quantmodule_output = self.quantmodule(features)
        if self.residual_connection:
            quantmodule_output = quantmodule_output + self.residual_connection(features)
        if self.use_labels and return_classification:
            predictions = self.classification_module(features)
            return self.output_module(quantmodule_output), predictions
        else:
            return self.output_module(quantmodule_output)


class DLQuantification:
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
    :param random_seed: Seed to make results reproducible. This net needs to generate the bags so the seed is important.
    :type random_seed: int
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
        random_seed,
        feature_extraction_module,
        quantmodule,
        linear_sizes,
        dropout,
        batch_size: int,
        bag_generator: BaseBagGenerator,
        val_bag_generator: BaseBagGenerator = None,
        test_bag_generator: BaseBagGenerator = None,
        gradient_accumulation: int = None,
        batch_size_fe=None,
        optimizer_class=torch.optim.AdamW,
        weight_decay: float = 0,
        lr_factor=0.1,
        val_split=0,
        quant_loss=torch.nn.L1Loss(),
        epsilon=0,
        output_function="softmax",
        use_labels: bool = False,
        residual_connection: bool = False,
        use_labels_epochs=None,
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
        """Refer to the class documentation."""
        self.train_epochs = train_epochs
        self.initial_epoch = 0  # This number will change in case we load a checkpoint
        self.test_epochs = test_epochs
        self.n_classes = n_classes
        self.optimizer_class = optimizer_class
        self.optimizer = None
        self.start_lr = start_lr
        self.end_lr = end_lr
        if type(n_bags) is list or type(n_bags) is tuple:
            if len(n_bags) != 3:
                raise ValueError(
                    "If n_bags is tuple/list it should have 3 elements (n_bags_train,n_bags_val, \
                                  n_bags_test)"
                )
            self.n_bags_train = n_bags[0]
            self.n_bags_val = n_bags[1]
            self.n_bags_test = n_bags[2]
        elif type(n_bags) is int:
            self.n_bags_train = n_bags
            self.n_bags_val = n_bags
            self.n_bags_test = n_bags
        else:
            raise ValueError(
                "n_bags should be an int or a tuple with three elements (n_bags_train,n_bags_val, \
                              n_bags_test)"
            )
        if type(bag_size) is tuple:
            if len(n_bags) != 3:
                raise ValueError(
                    "If bag_size is tuple/list it should have 3 elements (bag_size_trian,bag_size_val, \
                                  bag_size_test)"
                )
            self.bag_size_train = bag_size[0]
            self.bag_size_val = bag_size[1]
            self.bag_size_test = bag_size[2]
        else:
            self.bag_size_train = self.bag_size_val = self.bag_size_test = bag_size

        self.random_seed = random_seed
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.output_function = output_function
        self.lr_factor = lr_factor
        self.batch_size = batch_size
        self.bag_generator = bag_generator
        self.val_bag_generator = val_bag_generator
        self.test_bag_generator = test_bag_generator
        self.val_split = val_split
        self.gradient_accumulation = gradient_accumulation
        self.patience = patience
        self.quant_loss = quant_loss
        self.class_loss = F.cross_entropy
        self.use_labels = use_labels
        self.use_fp16 = use_fp16
        self.use_labels_epochs = use_labels_epochs
        self.residual_connection = residual_connection
        self.device = device
        self.save_model_path = save_model_path
        self.save_checkpoint_epochs = save_checkpoint_epochs
        if self.save_checkpoint_epochs is not None and self.save_model_path is None:
            raise ValueError("If you want to save model checkpoints, the save_model_path is needed")
        self.verbose = verbose
        self.epsilon = epsilon
        self.callback_epoch = callback_epoch
        self.dataset_name = dataset_name
        self.checkpoint_loaded = False
        self.best_error = float("inf")  # Highest value. We want to store the best error during the epochs
        # make results reproducible
        torch.manual_seed(random_seed)

        self.log_samples = log_samples
        if tensorboard_dir is not None:
            timestamp = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
            self.writer = SummaryWriter(os.path.join(tensorboard_dir, self.dataset_name + "_" + timestamp))
            self.writer.add_text("Hyperparameters", str(vars(self)))
            self.tensorboard = True
        else:
            self.tensorboard = False
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.login()
            wandb.init(
                # Set the project where this run will be logged
                project=dataset_name,
                name=wandb_experiment_name,
                save_code=True,
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                # Track hyperparameters and run metadata
                config={
                    "start_lr_rate": start_lr,
                    "end_lr_rate": end_lr,
                    # TODO: Log information for quant module
                    "n_classes": n_classes,
                    # "dropout": dropout,
                    "loss_func": quant_loss,
                    "batch_size": batch_size,
                    "gradient_accumulation": gradient_accumulation,
                    "patiente": patience,
                    "output_function": output_function,
                    "optimizer": optimizer_class,
                    "bag_size": bag_size,
                    "n_bags": n_bags,
                    "residual_connection": residual_connection,
                    "linear_sizes": linear_sizes,
                    "fe_output_size": feature_extraction_module.output_size,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                },
            )

        # Init the model
        self.model = DLQuantificationModule(
            n_classes=n_classes,
            dropout=dropout,
            feature_extraction_module=feature_extraction_module,
            quantmodule=quantmodule,
            linear_sizes=linear_sizes,
            use_labels=use_labels,
            output_function=output_function,
            residual_connection=residual_connection,
            batch_size_fe=batch_size_fe,
        )
        self.model.to(self.device)

        # Check if the model is already trained
        if self.save_model_path is not None and os.path.isfile(self.save_model_path):
            if self.verbose > 0:
                print("Model %s exists. Loading weights ..." % self.save_model_path)
            self.model.load_state_dict(torch.load(self.save_model_path))

    def __move_data_device(self, data):
        """Move the data to the device used (usually GPU).

        If the data is not a Tensor will convert it to Tensor with  the appropiate type.

        :return: Tensor with the data in the device
        :rtype: Tensor
        :param data: Data to move to the device
        :type data: np.array or torch.Tensor
        """
        if isinstance(data, dict):  # case for the bert dataset. We suppose the data is already in the device
            return {key: val.to(self.device) for key, val in data.items()}
        if torch.is_tensor(data):
            return data.to(self.device)
        else:
            if data.dtype == "float64":
                return torch.tensor(data).float().to(self.device)
            elif data.dtype == "int32" or data.dtype == "int64":
                return torch.tensor(data).long().to(self.device)
            else:
                return torch.tensor(data).to(self.device)

    def __log(self, epoch, iter, loss, p, p_hat, train):
        if epoch == 0 and iter == 0:
            self.file_log_train = self.dataset_name + "_train_samples.csv"
            self.file_log_val = self.dataset_name + "_val_samples.csv"
            if train and os.path.exists(self.file_log_train):
                os.remove(self.file_log_train)
            if not train and os.path.exists(self.file_log_val):
                os.remove(self.file_log_val)

        if iter == 0:
            column_names = ["Epoch", "Sample", "Loss"]
            for i in range(self.n_classes):
                column_names.append("p_" + str(i))
                column_names.append("p_hat_" + str(i))
            self.log_dataframe_train = pd.DataFrame(
                index=np.arange(self.n_bags_train), columns=column_names, dtype=float
            )
            self.log_dataframe_val = pd.DataFrame(index=np.arange(self.n_bags_val), columns=column_names, dtype=float)
        if train:
            self.log_dataframe_train.iloc[iter, 0] = epoch
            self.log_dataframe_train.iloc[iter, 1] = iter
            self.log_dataframe_train.iloc[iter, 2] = loss
            self.log_dataframe_train.iloc[iter, 3:] = np.concatenate(
                list(zip(p.cpu().detach().numpy(), p_hat.cpu().detach().numpy()))
            )
        else:
            self.log_dataframe_val.iloc[iter, 0] = epoch
            self.log_dataframe_val.iloc[iter, 1] = iter
            self.log_dataframe_val.iloc[iter, 2] = loss
            self.log_dataframe_val.iloc[iter, 3:] = np.concatenate(list(zip(p.cpu().numpy(), p_hat.cpu().numpy())))

        if train and iter == self.n_bags_train - 1:
            self.log_dataframe_train = self.log_dataframe_train.astype({"Epoch": int, "Sample": int})
            self.log_dataframe_train.to_csv(
                self.file_log_train, mode="a", index=False, float_format="%.3f", header=(epoch == 0)
            )
        if not train and iter == self.n_bags_val - 1:
            self.log_dataframe_val = self.log_dataframe_val.astype({"Epoch": int, "Sample": int})
            self.log_dataframe_val.to_csv(
                self.file_log_val, mode="a", index=False, float_format="%.3f", header=(epoch == 0)
            )

    def __compute_validation_loss(self, val_dataloader, epoch):
        """
        Compute loss over the validation dataset.

        :return: If not using labels, the loss returned is the quantification loss, if using labels, we return
                 [total_loss, quant_loss, class_loss]
        :rtype: list
        :param val_dataloader: A torch dataloader with the validation data
        :type val_dataloader: torch.DataLoader
        """

        start_time = time.time()

        if self.verbose > 0:
            print("[{}] Starting validation...".format(self.dataset_name))

        val_total_loss = 0
        if self.use_labels:
            val_quant_loss = 0
            val_class_loss = 0

        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(val_dataloader):
                X_batch = self.__move_data_device(batch["x"])
                num_samples_batch = X_batch[next(iter(X_batch))].shape[0] if type(X_batch) is dict else X_batch.shape[0]
                batch["samples"] = torch.arange(i * self.batch_size, (i * self.batch_size) + num_samples_batch)
                if self.use_labels:
                    y_batch = self.__move_data_device(batch["y"])

                if self.tensorboard:  # Logging to tensorboard
                    if i == (self.n_bags_val // 2):  # Just log one sample each epoch
                        if self.use_labels:
                            self.writer.add_histogram("Val sample prevalences", y_batch, epoch, bins=self.n_classes)
                        self.writer.add_text("Val sample prevalences", str(val_dataloader.batch_sampler.p[i, :]), epoch)

                if self.use_labels:
                    p_hat, predictions = self.model.forward(X_batch, return_classification=True)
                    quant_loss = self.quant_loss(
                        val_dataloader.batch_sampler.p[batch["samples"], :].to(self.device), p_hat
                    )
                    class_loss = self.class_loss(
                        predictions.view(predictions.shape[0] * predictions.shape[1], -1),
                        y_batch.view(y_batch.shape[0] * y_batch.shape[1]),
                    )
                    total_loss = quant_loss + class_loss
                else:
                    p_hat = self.model.forward(X_batch)
                    total_loss = self.quant_loss(
                        val_dataloader.batch_sampler.p[batch["samples"], :].to(self.device), p_hat
                    )

                val_total_loss += total_loss.item()
                if self.use_labels:
                    val_quant_loss += quant_loss.item()
                    val_class_loss += class_loss.item()

                if self.log_samples:
                    self.__log(
                        epoch,
                        i,
                        total_loss.item(),
                        val_dataloader.batch_sampler.p["batch_samples", :],
                        p_hat,
                        train=False,
                    )

            val_total_loss /= len(val_dataloader)
            if self.use_labels:
                val_quant_loss /= len(val_dataloader)
                val_class_loss /= len(val_dataloader)

        end_time = time.time()
        elapsed = end_time - start_time
        print("[{}] Elapsed validation time:{:.2f}s".format(self.dataset_name, elapsed))

        if self.use_labels:
            return val_total_loss, val_quant_loss, val_class_loss
        else:
            return val_total_loss

    # I have found no standard way of getting the labels from a dataset
    def __get_dataset_targets(self, dataset: Dataset):
        if isinstance(dataset, TensorDataset):
            if len(dataset.tensors) == 1:
                return None
            else:
                return dataset.tensors[1]
        elif isinstance(dataset, Subset):
            # TODO: here we should also check if we have two dimensions or only one
            return torch.as_tensor(dataset.dataset.targets)[dataset.indices]
        else:
            if type(dataset.targets) == list:
                return torch.LongTensor(dataset.targets).to(self.device)
            else:
                return dataset.targets

    def __compute_train_validation_split(self, dataset: Dataset, val_split, random_seed):
        if isinstance(val_split, int):
            if val_split == 0:
                return None
            else:  # Take this number of examples as validation
                val_size = val_split
                train_size = len(dataset) - val_size
                return torch.utils.data.random_split(
                    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed)
                )

        # In this case (float) this is a percentage
        if isinstance(val_split, float):
            val_size = round(len(dataset) * val_split)
            train_size = len(dataset) - val_size
            return torch.utils.data.random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed)
            )

        if isinstance(val_split, tuple):
            train_set = torch.utils.data.Subset(dataset, val_split[0])
            val_set = torch.utils.data.Subset(dataset, val_split[1])
            return train_set, val_set

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay, momentum=0.9
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=self.patience, factor=self.lr_factor, cooldown=0, verbose=True
        )
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.initial_epoch = checkpoint["epoch"] + 1
        self.best_error = checkpoint["best_error"]
        self.checkpoint_loaded = True

    def __getinputsize(self, input):
        if isinstance(input, torch.Tensor):
            return input.shape[0]
        elif isinstance(input, dict):
            return next(iter(input.values())).shape[0]
        else:
            raise ValueError("Cannot create batches on unknown type of input data")

    def __getinputdevice(self, input):
        if isinstance(input, torch.Tensor):
            return input.device
        elif isinstance(input, dict):
            return next(iter(input.values())).device
        else:
            raise ValueError("Cannot create batches on unknown type of input data")

    def __create_minibatches_input(self, input, batch_size_fe=8):
        """This is useful when we have a very big feature extraction layer and we want to create minibatches
        for this layer only."""
        inputsize = self.__getinputsize(input)

        if inputsize % batch_size_fe != 0:
            raise ValueError("Batch_size_fe must be divisible by input size")

        i = 0
        while (i + batch_size_fe) <= inputsize:
            if isinstance(input, torch.Tensor):
                yield input[
                    i : i + batch_size_fe,
                ]
            elif isinstance(input, dict):
                minibatch = dict()
                for key, value in input.items():
                    minibatch[key] = value[
                        i : i + batch_size_fe,
                    ]
                yield minibatch
            i += batch_size_fe

    def update_weights(self, batch_i, num_batches):
        if self.gradient_accumulation is None or self.gradient_accumulation == 1:
            return True

        if batch_i != 0 and batch_i % self.gradient_accumulation == 0:
            return True

        if batch_i == num_batches - 1:
            return True

        return False

    def fit(self, dataset: Dataset, val_dataset: Dataset = None):
        """
        Fits the model to the dataset.

        :return: Best loss achieved after finishing the training process
        :rtype: float
        :param dataset: torch.Dataset class with the training and validation data. The split will be done inside the
                        fit method based on the val_split parameter.
        :type dataset: torch.Dataset
        :param val_dataset: torch.Dataset we can choose to pass a validation dataset. In this case, val_split does not
                            make sense. This will be the split.
        :type dataset: torch.Dataset
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("Error: dataset must be a torch Dataset class")

        self.targets = self.__get_dataset_targets(dataset)

        if val_dataset is None:
            # split training into train and validation
            split = self.__compute_train_validation_split(dataset, self.val_split, self.random_seed)
            if split is None:
                self.is_val_set = False
                train_set = dataset
                val_set = dataset
                train_set_targets, val_set_targets = self.targets, self.targets
                if self.verbose > 0:
                    print("Using training set as validation set for early stopping")
            else:
                self.is_val_set = True
                train_set, val_set = split[0], split[1]
                # TODO: if there is no labels in the training this will fail. This is because current implementation of
                # the bag sampler requires labels.
                if self.targets is not None:
                    train_set_targets, val_set_targets = self.targets[train_set.indices], self.targets[val_set.indices]
                else:
                    train_set_targets = val_set_targets = None
                if self.verbose > 0:
                    print(
                        "Spliting {} examples in training set [training: {}, validation: {}]".format(
                            len(dataset), len(train_set), len(val_set)
                        )
                    )
        else:
            if self.val_split is not None:
                print("Ignoring val_split as train and val datasets where given to fit function")
            self.is_val_set = True
            train_set, val_set = dataset, val_dataset
            if self.targets is not None:
                train_set_targets, val_set_targets = self.targets, self.__get_dataset_targets(val_dataset)
            else:
                train_set_targets = val_set_targets = None
            if self.verbose > 0:
                print(
                    "We have two datasets: training {} examples. validation {} examples.]".format(
                        len(train_set), len(val_set)
                    )
                )

        if self.verbose > 0:
            print("Using device {}".format(self.device))

        # Create the dataloader with custom batch_sampling
        train_batch_sampler = BagSampler(
            self.bag_generator,
            n_bags=self.n_bags_train,
            bag_size=self.bag_size_train,
            batch_size=self.batch_size,
            targets=train_set_targets,
        )

        train_dataloader = DataLoader(
            train_set,
            batch_sampler=train_batch_sampler,
            collate_fn=partial(batch_collate_fn, bag_size=self.bag_size_train),
            num_workers=self.num_workers,
        )
        if self.is_val_set:
            val_batch_sampler = BagSampler(
                self.val_bag_generator,
                n_bags=self.n_bags_val,
                bag_size=self.bag_size_val,
                batch_size=self.batch_size,
                targets=val_set_targets,
            )
            val_dataloader = DataLoader(
                val_set,
                batch_sampler=val_batch_sampler,
                collate_fn=partial(batch_collate_fn, bag_size=self.bag_size_val),
                num_workers=self.num_workers,
            )

        if not self.checkpoint_loaded:  # if a checkpoint is loaded, we already have the optimizer created
            self.optimizer = self.optimizer_class(
                self.model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=self.patience, factor=self.lr_factor, cooldown=0, verbose=True
            )

        # fp16
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
        if self.use_wandb:
            wandb.watch(self.model, log="all")

        for epoch in range(self.initial_epoch, self.train_epochs):
            if self.use_labels_epochs is not None and epoch > self.use_labels_epochs:
                self.use_labels = False

            start_epoch = time.time()
            if self.verbose > 0:
                print("[{}] Starting epoch {}...".format(self.dataset_name, epoch))

            self.model.train()
            train_total_loss = batch_total_loss = train_quant_loss = batch_quant_loss = 0
            if self.use_labels:
                train_class_loss = batch_class_loss = 0

            # Log learning rate
            if self.tensorboard:
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
            if self.use_wandb:
                wandb.log({"lr": self.optimizer.param_groups[0]["lr"]}, step=epoch)

            # Loop for processing the batch
            for batch_i, (batch) in enumerate(train_dataloader):
                # store the indexes of the bags that we are processing in this minibatch
                X_batch = self.__move_data_device(batch["x"])
                num_samples_batch = X_batch[next(iter(X_batch))].shape[0] if type(X_batch) is dict else X_batch.shape[0]
                batch["samples"] = torch.arange(
                    batch_i * self.batch_size, (batch_i * self.batch_size) + num_samples_batch
                )
                if self.use_labels:
                    y_batch = self.__move_data_device(batch["y"])

                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    if self.use_labels:
                        if self.tensorboard:
                            if batch_i == 0 and epoch == 0:
                                self.writer.add_graph(self.model, X_batch)
                        p_hat, predictions = self.model.forward(X_batch, return_classification=True)
                        quant_loss = self.quant_loss(
                            train_dataloader.batch_sampler.p[batch["samples"], :].to(self.device).float(), p_hat
                        )
                        # Check all at once (all the examples from all the samples)
                        class_loss = self.class_loss(
                            predictions.view(predictions.shape[0] * predictions.shape[1], -1),
                            y_batch.view(y_batch.shape[0] * y_batch.shape[1]),
                        )
                        total_loss = quant_loss + class_loss
                    else:
                        p_hat = self.model.forward(X_batch)
                        quant_loss = self.quant_loss(
                            train_dataloader.batch_sampler.p[batch["samples"], :].to(self.device).float(), p_hat
                        )
                        total_loss = quant_loss

                # Log a bag to check that everything is ok
                if self.tensorboard:
                    if batch_i == (self.n_bags_train // 2):
                        if self.use_labels:
                            self.writer.add_histogram(
                                "Training sample prevalences", y_batch, epoch, bins=self.n_classes
                            )
                        self.writer.add_text(
                            "Training sample prevalences", str(train_dataloader.batch_sampler.p[batch_i, :]), epoch
                        )

                if self.log_samples:
                    self.__log(
                        epoch,
                        batch_i,
                        total_loss.item(),
                        train_dataloader.batch_sampler.p[batch_i, :],
                        p_hat,
                        train=True,
                    )

                # Store the losses for logging purposes
                train_total_loss += total_loss.item()
                batch_total_loss += total_loss.item()
                train_quant_loss += quant_loss.item()
                batch_quant_loss += quant_loss.item()
                # If we are using labels, compute the class loss also
                if self.use_labels:
                    train_class_loss += class_loss.item()
                    batch_class_loss += class_loss.item()

                if abs(quant_loss.item()) >= self.epsilon:  # Idea. Taken from SVR.
                    # We compute the gradients from the total loss (could be only quantification loss or
                    # class+quant loss)
                    scaler.scale(total_loss).backward()
                else:
                    if self.verbose > 0:
                        print(
                            "Not updating weights as loss {} is smaller than epsilon {}".format(
                                total_loss, self.epsilon
                            )
                        )
                if self.update_weights(batch_i, len(train_dataloader)):
                    if self.verbose >= 20:
                        print("Updating weights batch = %d of %d" % (batch_i, len(train_dataloader)))
                    scaler.step(self.optimizer)
                    scaler.update()
                    # self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.tensorboard:
                        if self.use_labels:
                            self.writer.add_scalar(
                                "TRAIN/class loss/batch", batch_class_loss / num_samples_batch, batch_i
                            )
                        self.writer.add_scalar("TRAIN/quant loss/batch", batch_quant_loss / num_samples_batch, batch_i)
                        self.writer.add_scalar("TRAIN/total loss/batch", batch_total_loss / num_samples_batch, batch_i)
                        self.writer.flush()

            train_total_loss /= len(train_dataloader)
            train_quant_loss /= len(train_dataloader)
            if self.use_labels:
                train_class_loss /= len(train_dataloader)

            end_epoch = time.time()
            elapsed = end_epoch - start_epoch
            print("[{}] Elapsed training time:{:.2f}s".format(self.dataset_name, elapsed))

            # Compute validation losses after finishing the epoch
            if self.is_val_set:
                if self.use_labels:
                    val_total_loss, val_quant_loss, val_class_loss = self.__compute_validation_loss(
                        val_dataloader, epoch
                    )
                else:
                    val_quant_loss = self.__compute_validation_loss(val_dataloader, epoch)
                    val_total_loss = val_quant_loss
            else:
                val_total_loss = train_total_loss
                val_quant_loss = train_quant_loss
                if self.use_labels:
                    val_class_loss = train_class_loss

            if self.callback_epoch is not None:
                self.callback_epoch(val_total_loss, epoch)

            if self.tensorboard:
                if self.use_labels:
                    self.writer.add_scalar("TRAIN/class loss/epoch", train_class_loss, epoch)
                    self.writer.add_scalar("VAL/class loss/epoch", val_class_loss, epoch)
                self.writer.add_scalar("TRAIN/quant loss/epoch", train_quant_loss, epoch)
                self.writer.add_scalar("VAL/quant loss/epoch", val_quant_loss, epoch)
                self.writer.add_scalar("VAL/total loss/epoch", val_total_loss, epoch)
                self.writer.add_scalar("TRAIN/total loss/epoch", train_total_loss, epoch)
                self.writer.flush()

            if self.use_wandb:
                wandb.log({"loss/train": train_total_loss, "loss/val": val_total_loss})

            if self.verbose > 0:
                if self.use_labels:
                    print(
                        "[{}] [Epoch={:03d}] Train Loss=[TotalLoss={:.5f};QuantLoss={:.5f};ClassLoss={:.5f}]. Val loss=[TotalLoss={:.5f};QuantLoss={:.5f}; ClassLoss={:.5f}]".format(
                            self.dataset_name,
                            epoch,
                            train_total_loss,
                            train_quant_loss,
                            train_class_loss,
                            val_total_loss,
                            val_quant_loss,
                            val_class_loss,
                        ),
                        end="",
                    )
                else:
                    print(
                        "[{}] [Epoch={:03d}] Train Loss=[QuantLoss={:.5f}]. Val loss = [QuantLoss={:.5f}]".format(
                            self.dataset_name, epoch, train_total_loss, val_total_loss
                        ),
                        end="",
                    )
            # Save the best model up to this moment. #TODO: Maybe we want to monitor only the quant loss??
            if val_quant_loss < self.best_error:
                self.best_error = val_quant_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                if self.verbose > 0:
                    print("[saved best model in this epoch]", end="")

            print("")

            self.scheduler.step(val_total_loss)

            if self.optimizer.param_groups[0]["lr"] < self.end_lr:
                if self.verbose > 0:
                    print("[{}] Early stopping in epoch {}!".format(self.dataset_name, epoch))
                break

            if self.save_checkpoint_epochs is not None and epoch != 0 and (epoch % self.save_checkpoint_epochs) == 0:
                print("Saving a checkpoint of the model...")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "best_error": self.best_error,
                    },
                    self.save_model_path + ".ckpt",
                )

        if self.verbose > 0:
            print("[{}] Restoring best model...".format(self.dataset_name))
        self.model.load_state_dict(self.best_model)
        if self.save_model_path is not None:
            torch.save(self.model.state_dict(), self.save_model_path)
        if self.tensorboard:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()
        return self.best_error

    def predict(self, dataset):
        """Make predictions for a dataset.

        Args:
            dataset (torch.Dataset): Input dataset for the prediction

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            if self.bag_size_test >= len(dataset):
                if self.verbose >= 10:
                    print("Not generating bags for prediction as bag_size is equal or bigger than len(test_dataset)")
                test_dataloader = DataLoader(dataset, batch_size=self.bag_size_test, num_workers=self.num_workers)
                X = next(iter(test_dataloader))[0]
                X = X.unsqueeze(0)  # The network is prepared to work only with batches of samples
                X = self.__move_data_device(X)
                self.model.eval()
                return self.model.forward(X).flatten().cpu().detach().numpy()
            else:
                test_batch_sampler = BagSampler(
                    self.test_bag_generator, n_bags=self.n_bags_test, bag_size=self.bag_size_test
                )
                test_dataloader = DataLoader(dataset, batch_sampler=test_batch_sampler, num_workers=self.num_workers)
                predictions = torch.zeros((self.n_bags_test * self.test_epochs, self.n_classes), device=self.device)
                for epoch in range(self.test_epochs):
                    start_epoch = time.time()
                    if self.verbose > 10:
                        print("[{}] Starting testing epoch {}... ".format(self.dataset_name, epoch), end="")
                    self.model.eval()
                    for i, (X_bag, *_) in enumerate(test_dataloader):
                        X_bag = self.__move_data_device(X_bag)
                        predictions[(epoch * self.n_bags_test) + i, :] = self.model.forward(X_bag).flatten()
                end_epoch = time.time()
                elapsed = end_epoch - start_epoch
                print("[Time:{:.2f}s]".format(elapsed), end="")
                print("done.")

            return torch.mean(predictions, axis=0).cpu().detach().numpy()
