# AMP公式

## 公式(7)推导

公式(7)是：

$$
|C|^2 d = (C \cdot \text{Diag}(d) \cdot C^H)D1
$$

其中：

* $C$ 是一个复数矩阵。
* $d$ 是一个向量。
* $\text{Diag}(d)$ 是以向量 $d$ 为对角线元素的对角矩阵。
* $C^H$ 是 $C$ 的共轭转置矩阵。
* $D1$ 表示从矩阵 $(C \cdot \text{Diag}(d) \cdot C^H)$ 中提取对角元素组成的向量。

### 1. 推导左边的表达式

首先，$|C|^2$ 表示矩阵 $C$ 每个元素的模平方：

$$
|C|^2 = C \cdot \overline{C}
$$

这里的 $\overline{C}$ 表示 $C$ 的共轭矩阵（每个元素取共轭），所以 $|C|^2$ 表示 $C$ 中每个元素的模平方。

现在，计算 $|C|^2 \cdot d$：

$$
(|C|^2 \cdot d)_i = \sum_{j} |C_{ij}|^2 d_j
$$

其中，$|C\_{ij}|^2 = C\_{ij} \overline{C\_{ij\}}$ 表示 $C$ 的 $ij$ 元素的模平方。

### 2. 推导右边的表达式

现在我们来看右边的表达式 $(C \cdot \text{Diag}(d) \cdot C^H)D1$：

1.  计算 $C \cdot \text{Diag}(d)$：

    $$
    (C \cdot \text{Diag}(d))_{ij} = C_{ij} \cdot d_j
    $$
2.  然后，计算 $(C \cdot \text{Diag}(d) \cdot C^H)$：

    $$
    (C \cdot \text{Diag}(d) \cdot C^H)_{ik} = \sum_{j} C_{ij} \cdot d_j \cdot \overline{C_{kj}}
    $$
3.  提取对角元素 $D1$：

    $$
    D1 = \text{diag}\left(C \cdot \text{Diag}(d) \cdot C^H\right) = \left[\sum_{j} C_{ij} \cdot d_j \cdot \overline{C_{ij}}\right]_{i}
    $$
4.  最终，右边的表达式为：

    $$
    (C \cdot \text{Diag}(d) \cdot C^H)D1 = \left[\sum_{j} C_{ij} \cdot d_j \cdot \overline{C_{ij}}\right]_{i} = \sum_{j} |C_{ij}|^2 d_j
    $$

### 3. 对比两边

通过上面的推导，可以看到：

$$
|C|^2 \cdot d = \left[\sum_{j} |C_{ij}|^2 d_j\right]_i = (C \cdot \text{Diag}(d) \cdot C^H)D1
$$

因此，文中公式(7)的推导是正确的。它表明通过矩阵操作获得的对角元素与矩阵模平方后的逐元素乘积是等价的。

***

**Key Concept**\
复数矩阵的模平方和共轭转置的运算关系。

**Key Concept Explanation**\
在信号处理中，复数矩阵的模平方涉及到矩阵的元素平方和操作，而矩阵的共轭转置与对角线提取也在算法推导中起到关键作用。公式(7)表明了如何利用这些性质简化复杂矩阵运算，尤其是在大规模信号处理系统中。
