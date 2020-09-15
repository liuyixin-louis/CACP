

time python3 /home/young/liuyixin/CAMC_disllter/multi-run.py\
     /home/young/liuyixin/CAMC_disllter/experiments/resnet20-ddpg-private amc.py \
          --arch=resnet20_cifar /home/dataset/cifar \
               --resume=/home/young/liuyixin/8.31/distiller/examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar\
                    --amc-protocol=mac-constrained --amc-action-range 0.05 1.0 \
                        --support-ratio=0.7 -p=50 \
                        --etes=0.075 --amc-ft-epochs=0 \
                            --amc-prune-pattern=channels \
                            --amc-prune-method=fm-reconstruction \
                                --amc-agent-algo=DDPG \
                                --amc-cfg=/home/young/liuyixin/CAMC_disllter/auto_compression_channels.yaml \
                                    --amc-rllib=hanlab -j=1 

