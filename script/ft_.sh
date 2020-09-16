time python finetune.py \
/home/dataset/cifar \
--scan-dir=/home/young/liuyixin/CAMC_disllter/experiments/resnet56_cifar   \
--arch=resnet56_cifar --lr=0.1 --vs=0 -p=50 --epochs=60 \
--compress=/home/young/liuyixin/CAMC_disllter/ft.yaml \
--deterministic