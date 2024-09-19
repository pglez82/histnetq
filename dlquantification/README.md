# HistNetQ
HistNetQ is a deep neural network for quantification.

### Installation steps
First, clone the project:
```
git clone https://github.com/pglez82/histnetq
```

The best way to install HistNetQ and the dependencies is using an Anaconda python distribution and a conda virtual enviroment.

Create a conda enviroment and activate it:
```sh
conda create --name histnet
conda activate histnet
```
Install the dependencies, that means, installing pytorch (with gpu support):

```bash
conda install pytorch==1.12.1 torchvision cudatoolkit=10.2 pandas tensorboard wandb scikit-learn==1.0.1 scipy==1.7.1 tqdm quadprog cvxpy -c pytorch
```
Actual version of cuda depends on your machine.

Add DLQuantification project to the conda enviroment (replace by the full path of the project in your machine):
```
conda develop pathtoproject #replace as neccesary
```

Once you have pytorch installed, you are ready to go. You can check that HistNet is working by executing the example script in histnet_test.py. That will train the network against the MNIST dataset.

```
python -m dlquantification.examples.histnet_example
```

## Getting started

Here there a basic example of how to use HistNetQ.
```python
import torch
import torchvision
import torchvision.transforms as transforms
from dlquantification.featureextraction.cnn import CNNFeatureExtractionModule
from dlquantification.histnet import HistNet
from dlquantification.utils.utils import AppBagGenerator

trainset = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())

device = torch.device('cuda:1')

fe = CNNFeatureExtractionModule(output_size=256)
histnet = HistNet(train_epochs=5000,test_epochs=1,start_lr=0.001,end_lr=0.000001,n_bags=500,bag_size=16,n_bins=8,random_seed = 2032,linear_sizes=[256],
                    feature_extraction_module=fe,bag_generator=APPBagGenerator(device=device),batch_size=16,quant_loss=torch.nn.L1Loss(),lr_factor=0.2, patience=20, dropout=0.05,epsilon=0,weight_decay=0,histogram='hard',use_labels=False,val_split=0.2,device=device,verbose=1,dataset_name="test_mnist")
histnet.fit(trainset)
```
This example will fit the network to the MNIST dataset. Check histnet/histnet_example.py.

## Parameter documentation

Check [doc/HistNet.md](doc/HistNet.md) for more information.
