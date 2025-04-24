# 总结

#### 🔁 与 AMP/LAMP/VAMP 的联系：

| 算法                  | 迭代结构             | 非线性函数 ηt(⋅)             | 是否有 Onsager 校正 |
| ------------------- | ---------------- | ----------------------- | -------------- |
| **AMP**             | (2.13) 结构        | 固定 soft-threshold       | ✅ 有            |
| **LAMP**            | 同 AMP            | ηt 为可训练参数               | ✅ 有            |
| **VAMP**            | 双阶段 AMP          | ηt 、η\~t 均可训练           | ✅ 有（双 Onsager） |
| **TISTA**           | 与 LISTA 类似       | ηt 为 soft + 可学习门控       | ✅ 有            |
| **DL-BP / DAMPNet** | BP 样式 message 更新 | 深度网络学习 message function | ❓ 结构依模型不同      |

***

&#x20;

<figure><img src="../.gitbook/assets/Screenshot 2025-04-24 at 12.37.09 pm.png" alt=""><figcaption></figcaption></figure>

### Training

some methods used to prevent overfitting concludes the section.

### Dataset

The training dataset is split in parts of equal size called batches before the start of the learning procedure. When the training phase starts. The algorithm iterates over epochs. For each epoch, a training step is repeated for every batch that compose the training dataset. A training step is divided into two stages:

&#x20;• forward pass computes the outputs $$\hat{x}^{1}$$, . . . ,  from the pairs input label $$\{  ( y^d, x^d )\}^D_{d=1}$$, where this time D stands for the size of the batch;&#x20;

• backward pass updates the values of  $$V_1$$, . . . ,  by computing their gradients with respect to the batch and then by applying gradient descent. T represents the number of layer of the network.

数据集\
在训练开始之前，训练数据集会被**划分为大小相等的多个批次（batches）**。训练阶段开始后，算法会**按轮次（epochs）进行迭代**。在每一轮（epoch）中，针对每一个 batch 会重复进行一次训练步骤。每一个训练步骤包含两个阶段：

*   **前向传播（forward pass）**：根据输入标签对 $${(y^{(d)}, x^{(d)})}_{d=1}^{D}$$

    &#x20;  计算输出 $$\hat{x}^{(1)}, \ldots, \hat{x}^{(D)}，$$ 其中 D 是当前 batch 的大小；
* **反向传播（backward pass）**：计算损失函数相对于网络中参数 $$V_1, \ldots, V_T$$  的梯度，并应用**梯度下降**进行参数更新。其中 TT 表示网络的层数。

每个样本由三类随机源生成：**信号 x** 、**信道矩阵 H** 和 **噪声 n** ：

* x 从调制星座中随机、均匀采样；
* H 根据所选信道模型结构进行采样；
* 噪声 n 通过 SNR 推导出的标准差 σ 生成；
* 每个样本的 SNR 在 \[18, 23] dB 区间内随机选取。

因此，每个 batch 中的样本是四元组形式：

&#x20;$$\{(y^{(d)}, H^{(d)}, \sigma^{(d)}, x^{(d)})\}_{d=1}^{D}$$

其中   $$x^{(d)}$$ 是待预测的标签（ground truth）。

本论文中的所有算法都是 **离线训练的**（offline training）：

* 首先在 **随机采样的 i.i.d. 高斯信道矩阵** 上训练；
* 然后在 **i.i.d. 高斯信道矩阵与 Kronecker 信道模型矩阵** 上测试；
* 还会在 Kronecker 信道矩阵上再训练一次，并在相同模型生成的不同矩阵上测试，以验证泛化能力。

生成的训练集会划分出 **25% 作为验证集**（validation set），用于 early stopping 和交叉验证（cross-validation）。

每一步训练中，batch 都是动态随机生成的，因为生成 batch 的开销远小于前向和反向传播。

所有深度学习相关算法均基于 **TensorFlow 2.0.0** 实现、训练与测试。不同的 MIMO 配置和调制阶数下会使用不同的训练过程。

为了**防止过拟合**，训练过程中使用以下策略：

* Early stopping（早停法）
* Dropout（随机丢弃）
* Cross-validation（交叉验证）

这些策略在下一节会详细说明，并且已应用于所有基于深度学习的算法。

训练参数：

* 训练周期：2000 epochs
* 优化器：**Adam**
* 学习率：0.001
* 每个 batch 大小：1000

Adam 优化器会对梯度及其平方进行如下迭代更新：

&#x20;$$m_e = \beta_1 m_{e-1} + (1 - \beta_1) \nabla_W^{(k)} f_{f',e}$$ ,   $$v_e = \beta_2 v_{e-1} + (1 - \beta_2) (\nabla_W^{(k)} f_{f',e})^2$$

其中 (1−β1) 、(1−β2) 分别代表一阶和二阶矩估计的学习率。
