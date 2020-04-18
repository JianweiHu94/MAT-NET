# MAT-NET
This repository contains the implementation of MAT-NET introduced in our IJCAI 2019 paper.
[MAT-Net: Medial Axis Transform Network for 3D Object Recognition](https://doi.org/10.24963/ijcai.2019/109).

If you use our code or models, please cite our paper.

        @inproceedings{ijcai2019-109,
        title     = {MAT-Net: Medial Axis Transform Network for 3D Object Recognition},
        author    = {Hu, Jianwei and Wang, Bin and Qian, Lihui and Pan, Yiling and Guo, Xiaohu and Liu, Lingjie and Wang, Wenping},
        booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI-19}},
        publisher = {International Joint Conferences on Artificial Intelligence Organization},  
        pages     = {774--781},
        year      = {2019},
        month     = {7}
        }

# Dataset and Pre-trained model
We repaired 83:2% of all 3D models in ModelNet40 and constructed a MAT data set, named ModelNet40-MAT.
You can download the dataset and pre-trained model files in
https://drive.google.com/drive/folders/1P9VFcaXf4xXvBARU-W1OK-D0whfABkyD?usp=sharing


# Enviroment
python 2.7
CUDA 8.0
tensorflow 1.4

# Training
python train.py

# Evaluation
python evaluate.py




