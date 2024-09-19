python train_mnist_cifar.py -t histnet_hard_ae_fashionmnist -n histnet -p ../parameters/histnet_hard_fashionmnist.json -d fashionmnist -l ae -c cuda:0  &
python train_mnist_cifar.py -t histnet_hard_rae_fashionmnist -n histnet -p ../parameters/histnet_hard_fashionmnist.json -d fashionmnist -l rae -c cuda:1  &
wait
python train_mnist_cifar.py -t histnet_hard_mse_fashionmnist -n histnet -p ../parameters/histnet_hard_fashionmnist.json -d fashionmnist -l mse -c cuda:0  &
python train_mnist_cifar.py -t settransformers_ae_fashionmnist -n settransformers -p ../parameters/settransformers_fashionmnist.json -d fashionmnist -l ae -c cuda:1  &
wait
python train_mnist_cifar.py -t settransformers_rae_fashionmnist -n settransformers -p ../parameters/settransformers_fashionmnist.json -d fashionmnist -l rae -c cuda:0  &
python train_mnist_cifar.py -t settransformers_mse_fashionmnist -n settransformers -p ../parameters/settransformers_fashionmnist.json -d fashionmnist -l mse -c cuda:1  &
wait
python train_mnist_cifar.py -t deepsets_avg_ae_fashionmnist -n deepsets -p ../parameters/deepsets_avg.json -d fashionmnist -l ae -c cuda:0  &
python train_mnist_cifar.py -t deepsets_avg_rae_fashionmnist -n deepsets -p ../parameters/deepsets_avg.json -d fashionmnist -l rae -c cuda:1  &
wait
python train_mnist_cifar.py -t deepsets_avg_mse_fashionmnist -n deepsets -p ../parameters/deepsets_avg.json -d fashionmnist -l mse -c cuda:0  &
python train_mnist_cifar.py -t deepsets_median_ae_fashionmnist -n deepsets -p ../parameters/deepsets_median.json -d fashionmnist -l ae -c cuda:1  &
wait
python train_mnist_cifar.py -t deepsets_median_rae_fashionmnist -n deepsets -p ../parameters/deepsets_median.json -d fashionmnist -l rae -c cuda:0  &
python train_mnist_cifar.py -t deepsets_median_mse_fashionmnist -n deepsets -p ../parameters/deepsets_median.json -d fashionmnist -l mse -c cuda:1  &
wait
python train_mnist_cifar.py -t deepsets_max_ae_fashionmnist -n deepsets -p ../parameters/deepsets_max.json -d fashionmnist -l ae -c cuda:0  &
python train_mnist_cifar.py -t deepsets_max_rae_fashionmnist -n deepsets -p ../parameters/deepsets_max.json -d fashionmnist -l rae -c cuda:1  &
wait
python train_mnist_cifar.py -t deepsets_max_mse_fashionmnist -n deepsets -p ../parameters/deepsets_max.json -d fashionmnist -l mse -c cuda:0  &
