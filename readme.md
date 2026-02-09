# 🚢 Titanic Survival Prediction (PyTorch Version)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Active-success)

这是一个基于 **PyTorch** 深度学习框架实现的 Kaggle 泰坦尼克号生存预测项目。

本项目不仅仅是一个简单的预测模型，更包含了一套完整的、工程化的机器学习工作流，涵盖了**数据清洗**、**特征工程**、**模型训练可视化**以及**自动化推理**。

## ✨ 核心特性

* **深度神经网络 (DNN)**: 采用自定义的 3 层全连接神经网络 (`TitanicNet`)，结合 `ReLU` 激活函数和 `Dropout` 层，有效防止过拟合。
* **鲁棒的预处理流水线**:
    * 自动处理缺失值（Age 中位数填充、Embarked 众数填充、Fare 补全）。
    * 智能特征提取：从姓名中提取身份头衔 (`Title`)，计算家庭规模 (`FamilySize`) 及是否独行 (`IsAlone`)。
    * **One-Hot 编码**: 对性别、舱位、登船港口等类别特征进行独热编码。
* **工程化设计**:
    * **训练/推理一致性**: 自动对齐训练集和测试集的特征列（自动补全测试集中缺失的 One-Hot 列），防止维度不匹配报错。
    * **模型持久化**: 训练脚本不仅保存模型参数，还保存了特征列信息和统计量 (`stats`)，确保推理时的预处理标准与训练时完全一致。

## 📂 项目结构

```text
├── data_preprocess.py   # 数据清洗与特征工程核心逻辑
├── dataset.py           # PyTorch Dataset 类定义
├── train.py             # 模型定义、训练循环与 Loss 可视化
├── test.py              # 模型加载、推理与结果生成 (submission.csv)
├── checking_missing.py  # 数据探索工具 (EDA)
├── train.csv            # (需自行下载) 训练数据
└── test.csv             # (需自行下载) 测试数据

```

## 🧠 模型架构

模型采用多层感知机 (MLP) 结构：

1. **Input Layer**: 动态适配预处理后的特征维度
2. **Hidden Layer 1**: 32 神经元 + ReLU + Dropout (0.2)
3. **Hidden Layer 2**: 8 神经元 + ReLU + Dropout (0.2)
4. **Output Layer**: 2 神经元 (二分类)

使用 `CrossEntropyLoss` 作为损失函数，`Adam` (lr=0.001) 作为优化器。

## 🚀 快速开始

### 1. 安装依赖

请确保你的环境安装了以下 Python 库：

```bash
pip install torch pandas numpy matplotlib scikit-learn

```

### 2. 准备数据

将 Kaggle 的 [Titanic 数据集](https://www.kaggle.com/c/titanic/data) (`train.csv` 和 `test.csv`) 放入项目根目录。

### 3. 训练模型

运行训练脚本：

```bash
python train.py

```

* 模型将进行 **1000 epochs** 的训练。
* 训练过程中会实时打印 Loss，并在结束后自动弹出 **Loss 变化曲线图**。
* 训练完成后，模型权重及统计信息将保存为 `titanic_model.pth`。

### 4. 生成预测结果

运行测试脚本：

```bash
python test.py

```

* 脚本会自动加载 `titanic_model.pth`。
* 对测试集进行与训练集完全相同的预处理。
* 生成 `submission.csv` 文件，可直接提交至 Kaggle 查看分数。

## 🤝 贡献

如果你有更好的特征工程想法或模型调优建议，欢迎提交 PR 或 Issue！

```

```
