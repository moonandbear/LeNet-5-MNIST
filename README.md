# MNIST 手写数字分类

该项目使用 TensorFlow 实现了一个卷积神经网络（CNN），用于分类 MNIST 数据集中的手写数字。

## 项目概述

该模型使用 TensorFlow 和 Keras 构建，旨在对 MNIST 数据集中的手写数字（0-9）进行分类。数据集包含 60,000 张训练图片和 10,000 张测试图片，每张图片的大小为 28x28 像素。模型使用卷积神经网络（CNN）进行图像分类，并使用动态学习率调度器来训练模型。

## 需求

- Python 3.6 或更高版本
- TensorFlow 2.x
- NumPy
- Matplotlib

你可以通过以下命令安装所需的依赖项：

```bash
pip install tensorflow numpy matplotlib
数据集
该模型使用 MNIST 数据集进行训练。MNIST 数据集包含 28x28 像素的灰度手写数字图像（0-9）。数据集会在代码执行时自动从 TensorFlow 的 datasets 模块中下载。

模型架构
该 CNN 模型使用以下层次结构构建：

Conv2D 层：6 个滤波器，5x5 的卷积核，ReLU 激活函数。

MaxPooling2D 层：2x2 的最大池化层。

Conv2D 层：16 个滤波器，5x5 的卷积核，ReLU 激活函数。

MaxPooling2D 层：2x2 的最大池化层。

Conv2D 层：120 个滤波器，5x5 的卷积核，ReLU 激活函数。

Conv2D 层：84 个滤波器，1x1 的卷积核，ReLU 激活函数。

Flatten 层：将 3D 特征图展平成 1D 向量。

Dense 层：输出 10 类数字的分类结果，使用 softmax 激活函数。

使用方法
1. 加载和准备数据
MNIST 数据集会在代码运行时自动下载，并进行预处理，包括特征缩放和标签的独热编码处理。

2. 创建并训练模型
你可以通过修改以下代码来选择不同的激活函数和池化方式：


activation_type = 'relu'  # 可以选择 'relu' 或 'sigmoid' 等
pooling_type = 'max'      # 可以选择 'max' 或 'avg'
model = create_model(activation=activation_type, pooling=pooling_type)
然后使用以下命令开始训练模型：


model.fit(train_db, epochs=20, validation_data=val_db, validation_freq=1, callbacks=[callback], initial_epoch=0,
          steps_per_epoch=len(train_db), validation_steps=len(val_db))
3. 模型评估
训练完成后，你可以使用模型对测试集进行评估，并查看分类准确率及其他指标。

学习率调度
该项目实现了一个学习率调度器，在训练过程中动态调整学习率。学习率会随着训练的进行而逐步衰减。


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1 * (epoch - 10))
结果
在 20 个周期（epochs）后，模型将输出验证集的准确率以及其他评估指标。你可以根据需要调整训练轮数和其他超参数。

