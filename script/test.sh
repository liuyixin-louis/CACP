python \
/home/young/liuyixin/CAMC_disllter/amc.py \
--arch resnet20_cifar \
/home/dataset/cifar \
--resume /home/young/liuyixin/8.31/distiller/examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar \
--amc-protocol mac-constrained \
--etes 0.075 \
--amc-ft-epochs 0 \
--amc-prune-pattern channels \
--amc-prune-method fm-reconstruction \
--amc-agent-algo DDPG \
--amc-cfg /home/young/liuyixin/8.31/distiller/examples/auto_compression/amc/auto_compression_channels.yaml \
--amc-rllib hanlab \
-j 4 \
--amc-heatup-episodes 300 \
--amc-training-episodes 1000 \
--out-dir /home/young/liuyixin/CAMC_disllter/logs \
--name resnet50-cifar-camc-300-1000 \
--deterministic 
