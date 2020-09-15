time python3 /home/young/liuyixin/CAMC_disllter/multi-run.py\
     /home/young/liuyixin/CAMC_disllter/experiments/resnet56-ddpg-private amc.py \
          --arch=resnet56_cifar /home/dataset/cifar \
               --state_dict=/home/young/liuyixin/CAMC_disllter/checkpoints/pytorch_resnet_cifar10/cifar10-resnet56-f5939a66.pth\
                    --amc-protocol=mac-constrained --amc-action-range 0.05 1.0 \
                        --support-ratio=0.7 -p=50 \
                        --etes=0.075 --amc-ft-epochs=0 \
                            --amc-prune-pattern=channels --amc-prune-method=fm-reconstruction \
                                --amc-agent-algo=DDPG --amc-cfg=/home/young/liuyixin/CAMC_disllter/auto_compression_channels.yaml \
                                    --amc-rllib=hanlab -j=1 