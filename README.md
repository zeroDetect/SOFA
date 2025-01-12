# SOFA
SOFA: Service-Oriented Fine-Grained Attack Traffic Detection with Meta Learning
![image](https://github.com/zeroDetect/SOFA/blob/main/img/framework.png)


## Introduction
In this paper, we present SOFA, a novel two-stage framework designed to address the challenges of fine-grained unknown attack detection under the constraints of class imbalance and limited labeled attack samples. By introducing service triplet-based aggregation and combining one-class learning with meta-learning principles, SOFA achieves precise detection of both known and unknown attacks. At a high level, SOFA consists of four components named Service Aggregation, One-class Model Training, Fewshot-Learning, and Fine-Grained Detection.

## Testbed
The implementation of our experiments was executed on a server running CentOS Linux 7. Our program was carried out on NVIDIA GeForce RTX 4090 GPUs. Python 3.7.12 was selected as the programming language, and the PyTorch framework was utilized to enhance the functionality and efficiency of our experimental procedures. 

## Running
1. Extract triplets for the first-stage, enter the ./extract_triplet/ folder, run with
```
python triplet_extract.py
```
2. Collect session features for each triplet, enter the ./feature_extract/ folder, run with
```
python sessionpcap2npz.py
```
3. To perform this test code, run with
```
pthon main.py
```
## Reference
1. [A Survey on Threat Hunting in Enterprise Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10216378) -  Boubakr Nour, Makan Pourzandi, Mourad Debbabi, IEEE Communications Surveys & Tutorials, 2023.
2. [Towards fine-grained unknown class detection against the open-set attack spectrum with variable legitimate traffic](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=10363&context=sis_research) -  Ziming Zhao, Zhaoxuan Li, et al., IEEE/ACM Transactions on Networking, 2024.
3. [Detecting unknown encrypted malicious traffic in real time via flow interaction graph analysis](https://arxiv.org/pdf/2301.13686) -  chuanpu Fu, Qi Li, Ke Xu，Proc. NDSS, 2023.
4. [Et-bert: A contextualized datagram representation with pre-training transformers for encrypted traffic classification](https://dl.acm.org/doi/pdf/10.1145/3485447.3512217) -  Xinjie Lin, Gang Xiong, Gaopeng Gou, Zhen Li, Junzheng Shi, Jing Yu, Proc. WWW, 2022.
