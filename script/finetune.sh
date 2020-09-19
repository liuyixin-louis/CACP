time python finetune.py \
/home/dataset/cifar \
--scan-dir=logs/resnet56-cifar-camc357-100-800-conditionalReward___2020.09.14-234808   \
                --arch=resnet56_cifar --lr=0.1 --vs=0 -p=50 --epochs=400 \
                --compress=ft.yaml \
                -j=1 --deterministic --processes=16