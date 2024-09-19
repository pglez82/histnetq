# Table of Contents

* [histnet.histnet](#histnet.histnet)
  * [HistNetworkModule](#histnet.histnet.HistNetworkModule)
    * [forward](#histnet.histnet.HistNetworkModule.forward)
  * [HistNet](#histnet.histnet.HistNet)
    * [fit](#histnet.histnet.HistNet.fit)
    * [predict](#histnet.histnet.HistNet.predict)

<a id="histnet.histnet"></a>

# histnet.histnet

<a id="histnet.histnet.HistNetworkModule"></a>

## HistNetworkModule Objects

```python
class HistNetworkModule(torch.nn.Module)
```

HistNet Pytorch module. This is the full network definition: feature extraction, histogram, and fully connected layers to do the quantification.

**Arguments**:

- `n_classes` (`int`): Number of classes to use in the problem
- `histogram` (`class`): Histogram layer. Check histograms module to see options available
- `dropout` (`float`): Dropout to use in the last linear layers, after the histogram
- `feature_extraction_module` (`class`): Feature extraction part of the network. Check feature extration module to see options available
- `linear_sizes` (`list`): List of ints with the sizes of the linear layers after the  histogram. Last layer is implicit and we do not need to specify it
- `use_labels` (`bool`): Extra connection for the labels

<a id="histnet.histnet.HistNetworkModule.forward"></a>

#### forward

```python
def forward(input, return_classification=False)
```

**Arguments**:

- `input` (`Tensor`): Tensor with a batch for a forward pass
- `return_classification` (`(bool, optional)`): If true, returns also the classification output. Defaults to False.

**Returns**:

`(Tensor,Tensor)`: Output probabilities and if return_classification, classification probabilities

<a id="histnet.histnet.HistNet"></a>

## HistNet Objects

```python
class HistNet()
```

Class for using the HistNet quantifier-

HistNet builds creates artificial samples with fixed size and learns from them. Every example in each sample goes through
the network and we build a histogram with all the examples in a sample. This is used in the quantification module where we use
this vector to quantify the sample.

**Arguments**:

- `train_epochs` (`int`): How many times to repeat the process of going over training data. Each epoch will train over n_bags samples.
- `test_epochs` (`int`): How many times to repeat the process over the testing data (returned prevalences are averaged).
- `start_lr` (`float`): Learning rate for the network (initial value).
- `end_lr` (`float`): Learning rate for the network. The value will be decreasing after a few epochs without improving (check patiente parameter).
- `n_classes` (`int`): Number of classes
- `optimizer_class` (`class`): torch.optim class to make the optimization. Example torch.optim.Adam
- `lr_factor` (`float`): Learning rate decrease factor after patience epochs have passed without improvement.
- `batch_size` (`int`): Update weights after this number of samples.
- `patience` (`int`): Number of epochs after which we will decrease the learning rate if there is no improvement.
- `n_bags` (`int or (int,int,int)`): How many artificial samples to build per epoch. If we get a single value this is used for training, val and test.
If a tuple with three values is provided it will used as (n_bags_train,n_bags_val,n_bags_test)
- `bag_size` (`int`): Number of examples per sample.
- `bag_generator` (`class`): Class that will be in charge of generating the samples.
- `n_bins` (`int`): Number of bins used to build the histogram.
- `random_seed` (`int`): Seed to make results reproducible. This net needs to generate the bags so the seed is important.
- `dropout` (`float`): Dropout to use in the network (avoid overfitting).
- `weight_decay` (`float`): L2 regularization for the model.
- `val_split` (`int, float or tuple`): By default we validate using the train data. If a split is given, we partition the data for using it as
validation and early stopping. We can receive the split in different ways:    1) float: percentage 
of data reserved for validation. 2) int: if 0, training set is used as validation. If any other number, this number of examples
                    will be used for validation. 3) tuple: if we get a tuple, this will be the specific indexes used for validation
- `quant_loss` (`function`): loss function to optimize in the quantification problem. Classification loss if use_labels=True is fixed (CrossEntropyLoss used)
- `epsilon` (`float`): If the error is less than this number, do not update the weights in this iteration.
- `feature_extraction_module` (`torch.Module`): Pytorch module with the feature extraction layers.
- `linear_sizes` (`tuple`): Tuple or list with the sizes of the linear layers used in the quantification module.
- `histogram` (`str`): Which histogram to use (sigmoid, soft, softrbf, hard)
- `quantiles` (`boolean`): If true, use a quantile version of the histogram.
- `use_labels` (`boolean`): If true, use the class labels to help fit the feature extraction module of the network. A mix of quant_loss + CrossEntropyLoss
will be used as the loss in this case.-
- `use_labels_epochs` (`int`): After this number of epochs, do not use the labels anymore. By default is use_labels is true, labels are going to be used for all the epochs.
- `device` (`torch.device`): Device to use for training/testing.
- `callback_epoch` (`function`): Function to call after each epoch. Useful to optimize with Optuna
- `save_model_path` (`file`): File to save the model when trained. We also load it if exists to skip training.
- `tensorboard_dir` (`str`): Path to a directory where to store tensorboard logs. We can explore them using tensorboard --logdir directory. By default no logs are saved.
- `log_samples` (`bool`): If true the network will log all the generated samples with p and p_hat and the loss (for training and validation)
- `verbose` (`int`): Verbose level.
- `dataset_name` (`str`): Only for logging purposes.

<a id="histnet.histnet.HistNet.fit"></a>

#### fit

```python
def fit(dataset: Dataset)
```

Fits the model to the dataset.

**Arguments**:

- `dataset` (`torch.Dataset`): torch.Dataset class with the training and validation data. The split will be done inside the
fit method based on the val_split parameter.

**Returns**:

`float`: Best loss achieved after finishing the training process

<a id="histnet.histnet.HistNet.predict"></a>

#### predict

```python
def predict(dataset)
```

Makes the prediction over each sample repeated for n epochs. Final result will be the average.

