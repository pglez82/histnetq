python train_lequa.py -t settransformers_T1A_ablation -m -n settransformers -p ./parameters/settransformers_T1A.json -f rff -d T1A -c cuda:0  &
python train_lequa.py -t deepsets_max_T1A_ablation -m -n deepsets -p ./parameters/deepsets_max.json -f rff -d T1A -c cuda:1  &
wait
python train_lequa.py -t deepsets_avg_T1A_ablation -m -n deepsets -p ./parameters/deepsets_avg.json -f rff  -d T1A -c cuda:0 &
python train_lequa.py -t deepsets_median_T1A_ablation -m -n deepsets -p ./parameters/deepsets_median.json -f rff  -d T1A -c cuda:1 &
wait
python train_lequa.py -t histnet_hard_T1A_ablation -m -n histnet -p ./parameters/histnet_hard_T1A.json -f rff -d T1A -c cuda:1 &
wait

