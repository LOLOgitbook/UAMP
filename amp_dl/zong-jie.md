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

