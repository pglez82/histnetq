### Paper experiments

The experiments comprises two different datasets:
- Fashion-MNIST (cifar_mnist folder)
- Lequa-T1A and Lequa-T1B

In order to run the main experiments, one should check the .sh files which automate everything. This files, will train the models.

The results notebook is in charge of generating latex tables from the results.

The baselines for Lequa are provided by the official competition repository. This is the [link](https://github.com/HLT-ISTI/QuaPy/tree/lequa2022/LeQua2022) for accesing them.

#### Fashion-MNIST
Create the directory for the output of the models:
```bash
cd experiments/paper/fashionmnist
mkdir savedmodels predictions results
```

Train the deep learning models with:
```bash
./train_fashionmnist.sh
```

Train the classifier needed for the traditional quantification methods:
```bash
python finetune_mnist_cifar.py -c cuda:0 -d fashionmnist
```

#### LeQua datasets
Download the lequa data and place it in the lequa directory (https://zenodo.org/record/6546188#.Y85gZafMKHs and https://zenodo.org/records/11661820).
Create the directory for the output of the models:
```bash
cd experiments/paper
mkdir savedmodels predictions results
```


Train the deep learning models with:
```bash
./train_lequa_T1A.sh
./train_lequa_T1B.sh
./train_lequa_T1.sh
./train_lequa_T2.sh
```

Check the results with help of the jupyter notebook ```results.ipynb```
