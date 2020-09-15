time python ft_.py \
/home/dataset/cifar \
--scan-dir=/home/young/liuyixin/CAMC_disllter/experiments/resnet56_cifar/amc0.3   \
--output-csv=ft_60epoch_results.csv \
--arch=resnet56_cifar --lr=0.1 --vs=0 -p=50 --epochs=2 \
--compress=/home/young/liuyixin/CAMC_disllter/ft.yaml \
-j=1 --deterministic --processes=16