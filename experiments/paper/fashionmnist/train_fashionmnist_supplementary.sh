python train_mnist_cifar.py -t histnet_soft_ae_fashionmnist -n histnet -p ../parameters/histnet_soft_fashionmnist.json -d fashionmnist -l ae -c cuda:0  &
python train_mnist_cifar.py -t histnet_soft_rae_fashionmnist -n histnet -p ../parameters/histnet_soft_fashionmnist.json -d fashionmnist -l rae -c cuda:1  &
wait
python train_mnist_cifar.py -t histnet_softrbf_ae_fashionmnist -n histnet -p ../parameters/histnet_softrbf_fashionmnist.json -d fashionmnist -l ae -c cuda:0  &
python train_mnist_cifar.py -t histnet_softrbf_rae_fashionmnist -n histnet -p ../parameters/histnet_softrbf_fashionmnist.json -d fashionmnist -l rae -c cuda:1  &
wait
python train_mnist_cifar.py -t histnet_sigmoid_ae_fashionmnist -n histnet -p ../parameters/histnet_sigmoid_fashionmnist.json -d fashionmnist -l ae -c cuda:0  &
python train_mnist_cifar.py -t histnet_sigmoid_rae_fashionmnist -n histnet -p ../parameters/histnet_sigmoid_fashionmnist.json -d fashionmnist -l rae -c cuda:1  &