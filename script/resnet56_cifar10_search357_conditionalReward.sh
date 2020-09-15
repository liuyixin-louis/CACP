python /home/young/liuyixin/CAMC_disllter/amc.py\
 --arch resnet56_cifar /home/dataset/cifar --amc-protocol mac-constrained-conditional-reward \
 --etes 0.075 --amc-ft-epochs 0 --amc-prune-pattern channels \
 --amc-prune-method fm-reconstruction --amc-agent-algo DDPG \
 --amc-cfg /home/young/liuyixin/CAMC_disllter/auto_compression_channels.yaml \
 --amc-rllib hanlab -j 4  \
 --support-ratio 0.3 0.5 0.7 \
 --amc-heatup-episodes 100 --amc-training-episodes 800 \
 --out-dir /home/young/liuyixin/CAMC_disllter/logs \
 --name resnet56-cifar-camc357-100-800-conditionalReward --deterministic --conditional --pretrained