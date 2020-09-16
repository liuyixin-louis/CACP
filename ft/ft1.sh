python finetune.py \
/home/dataset/cifar \
--scan-dir=ft/ft_ckpt/amc0.3   \
--arch=resnet56_cifar --lr=0.1 --vs=0 -p=50 --epochs=400 \
--compress=ft.yaml \
--deterministic