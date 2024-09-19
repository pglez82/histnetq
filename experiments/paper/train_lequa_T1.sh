python train_lequa.py -t settransformers_T1 -n settransformers -p ./parameters/settransformers_T1.json -f rff -d T1 -c cuda:0  &
python train_lequa.py -t deepsets_max_T1 -n deepsets -p ./parameters/deepsets_max.json -f rff -d T1 -c cuda:1  &
wait
python train_lequa.py -t deepsets_avg_T1 -n deepsets -p ./parameters/deepsets_avg.json -f rff  -d T1 -c cuda:0 &
python train_lequa.py -t deepsets_median_T1 -n deepsets -p ./parameters/deepsets_median.json -f rff  -d T1 -c cuda:1 &
wait
# python train_lequa.py -t histnet_sigmoid_T1 -n histnet -p ./parameters/histnet_sigmoid_T1.json -f rff  -d T1 -c cuda:0 &
# python train_lequa.py -t histnet_soft_T1 -n histnet -p ./parameters/histnet_soft_T1.json -f rff  -d T1 -c cuda:1 &
# wait
# python train_lequa.py -t histnet_softrbf_T1 -n histnet -p ./parameters/histnet_softrbf_T1.json -f rff  -d T1 -c cuda:0 &
# python train_lequa.py -t histnet_hard_T1 -n histnet -p ./parameters/histnet_hard_T1.json -f rff -d T1 -c cuda:1 &
# wait

