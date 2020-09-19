# Conditional Automated Channel Pruning for Deep Neural Networks （AAAI-21)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

This repo contains the PyTorch implementation for paper [**Conditional Automated Channel Pruning for Deep Neural Networks**](). 

![CACP](https://i.loli.net/2020/09/19/IJRTSM3GoFkbEfy.png)



## Dependencies

Current code base is tested under following environment:

1. Python 3.7.3
2. PyTorch 1.3.1
3. CIFAR-10 dataset

Using the following command to install the Dependencies:

```bash
pip install -r requirements.txt
```





## Testing CACP

Current code base supports the automated pruning of **Resnet56** on **CIFAR10**. The pruning of Resnet56 consists of 2 steps: **1. strategy search and export the pruned weights; 2. fine-tune from pruned weights**.

To conduct the full pruning procedure, follow the instructions below (results might vary a little from the paper due to different random seed):

1. **Strategy Search and Export the Pruned Weights**

```bash 
bash ./script/search_export_cacp.sh
```

Results may differ due to different random seed. The strategy (perserve rate for each layer of Resnet56) we found and reported in the paper is:

```bash
# 0.3
[0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.625, 0.625, 0.625, 0.8125, 0.8125, 0.78125, 0.75, 0.71875, 0.6875, 0.65625, 0.625, 0.609375, 0.5, 0.4375, 0.4375, 0.84375, 0.890625, 0.90625, 0.90625, 0.890625]
# 0.5
[0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.40625, 0.59375, 0.53125, 0.5625, 0.5625, 0.5625, 0.53125, 0.53125, 0.5, 0.359375, 0.484375, 0.453125, 0.46875, 0.4375, 0.40625, 0.390625, 0.375, 0.359375]
# 0.7
[0.5, 0.5, 0.4375, 0.4375, 0.4375, 0.4375, 0.375, 0.375, 0.375, 0.40625, 0.28125, 0.28125, 0.25, 0.25, 0.25, 0.25, 0.21875, 0.21875, 0.125, 0.171875, 0.15625, 0.171875, 0.1875, 0.203125, 0.234375, 0.265625, 0.328125]
```

Note: the checkpoint of best compressed models under different target rates will be automatically saved in the log folder.

2. **Fine-tune from Pruned Weights**

After searching and exporting, we need to fine-tune from the pruned weights. For example, we can fine-tune using RL-step learning rate for 400 epochs by running:

```bash
bash ./script/finetune.sh
```









## CACP Compressed Model

We also provide the models and weights compressed by our CACP method. We provide compressed Resnet56 on the CIAR-10 dataset in PyTorch.

Detailed statistics are as follows:

| Models            | Acc@1 before FT (%) | Acc@1 after FT (%) | MACs(M) | Params(K) |
| ----------------- | ------------------- | ------------------ | ------- | --------- |
| Resnet56-30%FLOPS | 22.46               | 93.98              | 38.17   | 246.21    |
| Resnet56-50%FLOPS | 60.60               | 93.84              | 63.25   | 475.55    |
| Resnet56-70%FLOPS | 77.94               | 93.13              | 88.90   | 660.83    |





## Reference

If you find the repo useful, please kindly cite our paper:

```tex
@inproceedings{CACP_AAAI-21,
  title={CACP: Conditional Channel Pruning for Automated Model Compression},
  author={Yixin Liu, Yong Guo, Zichang Liu, Haohua Liu, Jingjie Zhang,Zejun Chen, Jing Liu, Jian Chen},
  booktitle={AAAI-21 Student Abstract and Poster Program Thirty-Fifth Conference on Artificial Intelligence},
  year={2020},
  url = {}
}
```
