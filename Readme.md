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

To conduct the full pruning procedure, follow the instructions below:

1. **Strategy Search and Export the Pruned Weights**

```bash 
bash ./script/search_export_cacp.sh
```

Note: the checkpoint of best compressed models under different target rates will be automatically saved in the log folder.

2. **Fine-tune from Pruned Weights**

After searching and exporting, we need to fine-tune from the pruned weights. For example, we can fine-tune using RL-step learning rate for 400 epochs by running:

```bash
bash ./script/finetune.sh
```

The following table is the result we get (results might vary a little from the paper due to different random seed):

![image-20200919032710550](https://i.loli.net/2020/09/19/vYjW3gN9KduLBIH.png)







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
