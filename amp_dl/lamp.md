# LAMP

一、  VAMP &#x20;

VAMP (Vector Approximate Message Passing) 是一种 **迭代推理算法**，用于求解线性观测模型

y=Ax+n&#x20;

在每一轮迭代中，VAMP 进行两个关键步骤（模块）：

1. **线性模块**（Linear Estimation）\
   假设一个线性高斯模型，基于当前估计对信号进行线性滤波，输出 $$\mathbf{r}_t$$   ，估计误差方差 $$\tau_t$$ 。
2. **非线性模块**（Denoising Step）\
   假设信号 x 有某种稀疏先验（如 Bernoulli-Gaussian），对    $$\mathbf{r}_t$$   做 MMSE 推断：   $$\mathbf{x}_{t+1} = \mathbb{E}[\mathbf{x} | \mathbf{r}_t, \tau_t]$$  &#x20;

此时用到了 **已知的先验 p(x)**。

#### 🔧 二、Learned VAMP 网络（深度学习视角）

Learned VAMP 是一种 **将 VAMP 结构“展开”成神经网络** 的方法，每一层对应一次 VAMP 迭代：

1. 网络中每层的 **线性操作**   $$\mathbf{W}_t \mathbf{y} + \ldots$$     是可学习参数；
2.  非线性操作由神经网络（或参数化 shrinkage 函数）实现，如：   $$\mathbf{x}_{t+1} = \text{shrink}_{\boldsymbol{\theta}_t}(\mathbf{r}_t)$$   &#x20;

    &#x20;

    这些 shrink 函数中参数 θt 是**训练得到的**。

通过 **反向传播（Backpropagation）**，我们对所有层的参数 Wt,θt 做损失函数（如 NMSE）最小化，从而实现学习。

#### ✅ 三、为什么两者学到一样的东西？

* VAMP 推理算法中的非线性估计器已经是 **最优的 MMSE 推断器**，前提是先验已知；
* 在训练信号遵循 Bernoulli-Gaussian 等分布时：
  * backpropagation 学习出的参数和 VAMP 中推导的公式恰好一致；
  * 所以 Learned VAMP 网络实际上在**学习过程中自然逼近了 VAMP 理论给出的最优结构**。

&#x20;

> **当训练数据满足 VAMP 假设，backprop 学到的就是 VAMP 中的公式。**

***

&#x20;

&#x20;

1.

```python
prob = problems.bernoulli_gaussian_trial 
```

生成一个稀疏高斯信号恢复问题的数据集。包含验证集、训练集和初始估计集合。

通过bernoulli\_gaussian产生的

2\.

```python
layers = networks.build_LAMP(prob,T=6,shrink='bg',untied=False)
```

&#x20;生成eta和 theta 初始





















