# ChatbotBert
本项目基于huggingface的transformers库搭建Bert分类模型，构建一个简单的Bert聊天机器人

## Pipeline
[搭建思路]（个人学习总结）(/doc/搭建思路.md)

## Install
```
Python 3.10.5
pip install requirements.txt
```

## Usage
### 训练数据集
训练数据文件夹为chatterbot_corpus，里面包含了9种类别的日常话题
### 验证数据集
验证数据文件夹为val_data, 也包含了9种类别的日常话题
### 训练
```
python train.py --data_dir train_data --val_dir val_data --pretrained_model  'bert-base-uncased' --epochs 20 --classes 9
```
### 测试推理
