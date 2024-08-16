# AMP公式

## 《Approximate Message Passing with Unitary Transformation》 公式(7)推导

公式(7)是：

$$
|C|^2 d = (C \cdot \text{Diag}(d) \cdot C^H)D1
$$

其中：

* $$C$$ 是一个复数矩阵。
* $$d$$ 是一个向量。
* $$\text{Diag}(d)$$ 是以向量 $$d$$ 为对角线元素的对角矩阵。
* $$C^H$$ 是 $$C$$ 的共轭转置矩阵。
* $$D1$$ 表示从矩阵 $$(C \cdot \text{Diag}(d) \cdot C^H)$$ 中提取对角元素组成的向量。

### 1. 推导左边的表达式

首先，$$|C|^2$$ 表示矩阵 $$C$$ 每个元素的模平方：

$$
|C|^2 = C \cdot \overline{C}
$$

这里的 $$\overline{C}$$ 表示 $$C$$ 的共轭矩阵（每个元素取共轭），所以 $$|C|^2$$ 表示 $$C$$ 中每个元素的模平方。

现在，计算 $$|C|^2 \cdot d$$：

$$
(|C|^2 \cdot d)_i = \sum_{j} |C_{ij}|^2 d_j
$$

其中，$$|C_{ij}|^2 = C_{ij} \overline{C_{ij}}$$ 表示 $$C$$ 的 $$ij$$ 元素的模平方。

### 2. 推导右边的表达式

现在我们来看右边的表达式 $$(C \cdot \text{Diag}(d) \cdot C^H)D1$$：

1.  计算 $$C \cdot \text{Diag}(d)$$：

    $$
    (C \cdot \text{Diag}(d))_{ij} = C_{ij} \cdot d_j
    $$
2.  然后，计算 $$(C \cdot \text{Diag}(d) \cdot C^H)$$：

    $$
    (C \cdot \text{Diag}(d) \cdot C^H)_{ik} = \sum_{j} C_{ij} \cdot d_j \cdot \overline{C_{kj}}
    $$
3.  提取对角元素 $$D1$$：

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

#### 1. 矩阵运算与正交性质

假设 $$\mathbf{V}$$ 是一个 $$N \times M$$ 的矩阵，并且它的列是正交的，即 $$\mathbf{V}^T \mathbf{V} = \mathbf{I}$$（单位矩阵）。

对于对角矩阵 $$\mathbf{D}_N$$ 和 $$\mathbf{D}_M$$，它们分别是 $$N \times N$$ 和 $$M \times M$$ 的矩阵，且只有对角线元素非零。

#### 2. 计算 $$\mathbf{V}^T \mathbf{D}_N \mathbf{V}$$ 的期望

我们来分析 $$\mathbf{V}^T \mathbf{D}_N \mathbf{V}$$ 的期望：

$$
\mathbb{E}\left[\mathbf{V}^T \mathbf{D}_N \mathbf{V}\right] = \mathbb{E}\left[\mathbf{V}^T \text{diag}(d_1, d_2, \dots, d_N) \mathbf{V}\right]
$$

其中 $$\text{diag}(d_1, d_2, \dots, d_N)$$ 是对角矩阵 $$\mathbf{D}_N$$。

对正交矩阵 $$\mathbf{V}$$ 进行变换，相当于对对角线上的元素 $$d_i$$ 做了一个线性变换，使得它们混合到非对角元素中去。然而，由于 $$\mathbf{V}$$ 的正交性质，这种混合是均匀的，并且由于矩阵的对称性和线性变换的特性，对于大 $$N$$ 值（即大系统极限），每个元素的贡献是等价的。

因此，我们可以近似地写出：

$$
\mathbf{V}^T \mathbf{D}_N \mathbf{V} \approx \frac{1}{N} \text{tr}(\mathbf{D}_N) \mathbf{I}
$$

这个近似成立的原因是，$$\mathbf{V}$$ 中的每一列基本上是从正交矩阵的列中选取的，正交矩阵具有均匀分布的性质，从而在大系统中，每个对角元素的期望值都接近于对角线上的平均值。

#### 3. 对 $$\mathbf{V} \mathbf{D}_M \mathbf{V}^T$$ 进行类似推导

同理，对于 $$\mathbf{V} \mathbf{D}_M \mathbf{V}^T$$，可以进行类似的推导：

$$
\mathbb{E}\left[\mathbf{V} \mathbf{D}_M \mathbf{V}^T\right] = \mathbb{E}\left[\mathbf{V} \text{diag}(d_1', d_2', \dots, d_M') \mathbf{V}^T\right]
$$

由于 $$\mathbf{V}$$ 是正交的或者接近正交的矩阵，并且 $$N$$ 和 $$M$$ 足够大，那么：

$$
\mathbf{V} \mathbf{D}_M \mathbf{V}^T \approx \frac{1}{M} \text{tr}(\mathbf{D}_M) \mathbf{I}
$$

#### 4. 结论

在大系统假设下，由于正交矩阵的对称性和均匀分布性质，对于矩阵乘法 $$\mathbf{V}^T \mathbf{D}_N \mathbf{V}$$ 和 $$\mathbf{V} \mathbf{D}_M \mathbf{V}^T$$，可以将它们的结果近似为 $$\frac{1}{N} \text{tr}(\mathbf{D}_N) \mathbf{I}$$ 和 $$\frac{1}{M} \text{tr}(\mathbf{D}_M) \mathbf{I}$$。这种近似主要依赖于迹（即矩阵对角元素之和）的集中性，尤其是在矩阵维度很大时。

这个推导展示了为什么可以用迹的平均值来近似矩阵的全局行为，并且证明了大系统极限下这种近似的合理性。



《IMPROVED VARIANCE PREDICTIONS IN APPROXIMATE MESSAGE PASSING》的公式怎么

<figure><img src=".gitbook/assets/Screenshot 2024-08-16 at 1.36.47 pm.png" alt=""><figcaption></figcaption></figure>

#### 1. 假设条件

我们设定了以下条件：

* $$N = 3$$，即使用一个 $$3 \times 3$$ 的矩阵。
*   $$\mathbf{V}$$ 是一个正交矩阵：

    $$
    \mathbf{V} = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ 0 & 0 & 1 \end{pmatrix}
    $$
*   $$\mathbf{D}_N$$ 是一个对角矩阵：

    $$
    \mathbf{D}_N = \begin{pmatrix} 4 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{pmatrix}
    $$

#### 2. 计算 $$\mathbf{V}^T \mathbf{D}_N \mathbf{V}$$

通过矩阵运算，我们计算了：

1.  $$\mathbf{V}^T$$：

    $$
    \mathbf{V}^T = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ 0 & 0 & 1 \end{pmatrix}
    $$
2.  $$\mathbf{V}^T \mathbf{D}_N$$：

    $$
    \mathbf{V}^T \mathbf{D}_N = \begin{pmatrix} \frac{4}{\sqrt{2}} & -\frac{2}{\sqrt{2}} & 0 \\ \frac{4}{\sqrt{2}} & \frac{2}{\sqrt{2}} & 0 \\ 0 & 0 & 1 \end{pmatrix}
    $$
3.  $$\mathbf{V}^T \mathbf{D}_N \mathbf{V}$$：

    $$
    \mathbf{V}^T \mathbf{D}_N \mathbf{V} = \begin{pmatrix} \frac{4}{\sqrt{2}} & -\frac{2}{\sqrt{2}} & 0 \\ \frac{4}{\sqrt{2}} & \frac{2}{\sqrt{2}} & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} 3 & 1 & 0 \\ 1 & 3 & 0 \\ 0 & 0 & 1 \end{pmatrix}
    $$

#### 3. 计算 $$\frac{1}{N} \text{tr}(\mathbf{D}_N)$$

1.  计算 $$\mathbf{D}_N$$ 的迹：

    $$
    \text{tr}(\mathbf{D}_N) = 4 + 2 + 1 = 7
    $$
2.  计算 $$\frac{1}{N} \text{tr}(\mathbf{D}_N)$$：

    $$
    \frac{1}{3} \times 7 = \frac{7}{3} \approx 2.33
    $$

所以，近似的对角矩阵将是：

$$
\frac{7}{3} \mathbf{I} = \begin{pmatrix} \frac{7}{3} & 0 & 0 \\ 0 & \frac{7}{3} & 0 \\ 0 & 0 & \frac{7}{3} \end{pmatrix} \approx \begin{pmatrix} 2.33 & 0 & 0 \\ 0 & 2.33 & 0 \\ 0 & 0 & 2.33 \end{pmatrix}
$$

#### 4. 比较结果

* $$\mathbf{V}^T \mathbf{D}_N \mathbf{V}$$ 的对角元素是 $$\{3, 3, 1\}$$。
* $$\frac{1}{N} \text{tr}(\mathbf{D}_N)$$ 乘以单位矩阵的对角元素近似为 $$\{2.33, 2.33, 2.33\}$$。

#### 5. 结论

* 虽然具体数值上存在一定差异，尤其是在 $$N$$ 较小的情况下，但当 $$N$$ 逐渐增大时，这些对角元素的近似性会变得更加明显。非对角元素已经是零了（在理想的正交条件下），对角元素的平均值接近于 $$\frac{1}{N} \text{tr}(\mathbf{D}_N)$$。

这个简单的例子说明了在较大的维度 $$N$$ 下，$$\mathbf{V}^T \mathbf{D}_N \mathbf{V}$$ 的对角元素将逐渐接近 $$\frac{1}{N} \text{tr}(\mathbf{D}_N)$$ 的值，从而验证了迹近似在大系统中的合理性。

## 对于这个问题chatgpt的回答整理如下：

具体来说，在文章的第六节中提到的 $$\overline{\mathbf{V}}^T\mathbf{D}_N\overline{\mathbf{V}}$$ 被近似为 $$\frac{1}{N}\operatorname{tr}(\mathbf{D}_N)\mathbf{I}$$，其实现过程如下：

1. **Haar 分布和正交性**：$$\overline{\mathbf{V}}$$ 表示一个正交矩阵，其列向量从 Haar 分布中随机抽取。Haar 分布意味着 $$\overline{\mathbf{V}}$$ 在正交矩阵的空间中是均匀分布的。
2.  **迹近似**：对于大的 $$N$$，$$\overline{\mathbf{V}}^T\mathbf{D}_N\overline{\mathbf{V}}$$ 这个乘积在统计上表现为一个标量与单位矩阵的乘积，这个标量就是矩阵 $$\mathbf{D}_N$$ 对角线上元素的平均值。用数学表达式表示为：

    $$
    \overline{\mathbf{V}}^T\mathbf{D}_N\overline{\mathbf{V}} \approx \frac{1}{N}\operatorname{tr}(\mathbf{D}_N)\mathbf{I}
    $$

    其中 $$\operatorname{tr}(\mathbf{D}_N)$$ 是 $$\mathbf{D}_N$$ 的迹，即对角线上所有元素的和。
3. **大系统极限**：随着 $$N$$ 的增加，由 $$\overline{\mathbf{V}}$$ 引入的随机性由于大数定律的作用而趋于平均化，从而使得这个近似更加精确，最终将随机变化简化为由迹决定的确定性形式。
4. **直观理解**：关键的直观理解是，当 $$N$$ 足够大时，$$\mathbf{D}_N$$ 中个别元素的影响在正交变换 $$\overline{\mathbf{V}}$$ 下被平均化，使得乘积表现为一个缩放的单位矩阵。

这种技术在分析大随机矩阵时非常常用，特别是在信号处理和通信中的近似消息传递（AMP）及相关算法中。

假设我们有一个对角矩阵 $$\mathbf{D}_N$$，其中 $$N = 3$$，$$\mathbf{D}_N$$ 的对角元素分别为 $$[d_1, d_2, d_3]$$。这个矩阵可以表示为：

$$
\mathbf{D}_N = \begin{pmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{pmatrix}
$$

假设 $$\overline{\mathbf{V}}$$ 是一个 $$3 \times 3$$ 的正交矩阵，这意味着它的列向量是单位正交的，并且 $$\overline{\mathbf{V}}^T \overline{\mathbf{V}} = \mathbf{I}$$。

例如：

$$
\overline{\mathbf{V}} = \begin{pmatrix} v_{11} & v_{12} & v_{13} \\ v_{21} & v_{22} & v_{23} \\ v_{31} & v_{32} & v_{33} \end{pmatrix}
$$

我们现在考虑矩阵乘积 $$\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}}$$：

$$
\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}} = \begin{pmatrix} v_{11} & v_{21} & v_{31} \\ v_{12} & v_{22} & v_{32} \\ v_{13} & v_{23} & v_{33} \end{pmatrix} \begin{pmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{pmatrix} \begin{pmatrix} v_{11} & v_{12} & v_{13} \\ v_{21} & v_{22} & v_{23} \\ v_{31} & v_{32} & v_{33} \end{pmatrix}
$$

这个结果是一个 $$3 \times 3$$ 的矩阵，其中每个元素是一个涉及 $$v_{ij}$$ 和 $$d_i$$ 的组合。

在大系统假设下，即当 $$N$$ 很大时，$$\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}}$$ 的行为可以近似为：

$$
\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}} \approx \frac{1}{N} \operatorname{tr}(\mathbf{D}_N) \mathbf{I}
$$

其中 $$\operatorname{tr}(\mathbf{D}_N)$$ 是 $$\mathbf{D}_N$$ 的迹，即 $$d_1 + d_2 + d_3$$。

对于这个例子，$$\frac{1}{N} \operatorname{tr}(\mathbf{D}_N)$$ 可以计算为：

$$
\frac{1}{3} (d_1 + d_2 + d_3)
$$

因此，我们可以近似地说：

$$
\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}} \approx \frac{1}{3} (d_1 + d_2 + d_3) \mathbf{I}
$$

这里的 $$\mathbf{I}$$ 是 $$3 \times 3$$ 的单位矩阵：

$$
\mathbf{I} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

最终的近似结果是：

$$
\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}} \approx \frac{1}{3} (d_1 + d_2 + d_3) \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} \frac{1}{3}(d_1 + d_2 + d_3) & 0 & 0 \\ 0 & \frac{1}{3}(d_1 + d_2 + d_3) & 0 \\ 0 & 0 & \frac{1}{3}(d_1 + d_2 + d_3) \end{pmatrix}
$$

这个近似表明，原本复杂的矩阵乘积在大系统假设下可以简化为一个与迹相关的标量乘以单位矩阵。这种简化大大降低了计算复杂度，并且在 $$N$$ 足够大时是合理的。

<figure><img src=".gitbook/assets/Screenshot 2024-08-16 at 2.54.52 pm.png" alt=""><figcaption></figcaption></figure>

假设 (\mathbf{D}\_N) 是一个 (3 \times 3) 的对角矩阵：

$$
\mathbf{D}_N = \begin{pmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{pmatrix}
$$

(\overline{\mathbf{V\}}) 是一个 (3 \times 3) 的正交矩阵：

$$
\overline{\mathbf{V}} = \begin{pmatrix} v_{11} & v_{12} & v_{13} \\ v_{21} & v_{22} & v_{23} \\ v_{31} & v_{32} & v_{33} \end{pmatrix}
$$

首先，我们计算 (\mathbf{D}\_N \overline{\mathbf{V\}}) 的乘积：

$$
\mathbf{D}_N \overline{\mathbf{V}} = \begin{pmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{pmatrix} \begin{pmatrix} v_{11} & v_{12} & v_{13} \\ v_{21} & v_{22} & v_{23} \\ v_{31} & v_{32} & v_{33} \end{pmatrix} = \begin{pmatrix} d_1v_{11} & d_1v_{12} & d_1v_{13} \\ d_2v_{21} & d_2v_{22} & d_2v_{23} \\ d_3v_{31} & d_3v_{32} & d_3v_{33} \end{pmatrix}
$$

接下来计算 (\overline{\mathbf{V\}}^T \mathbf{D}\_N \overline{\mathbf{V\}})：

$$
\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}} = \begin{pmatrix} v_{11} & v_{21} & v_{31} \\ v_{12} & v_{22} & v_{32} \\ v_{13} & v_{23} & v_{33} \end{pmatrix} \begin{pmatrix} d_1v_{11} & d_1v_{12} & d_1v_{13} \\ d_2v_{21} & d_2v_{22} & d_2v_{23} \\ d_3v_{31} & d_3v_{32} & d_3v_{33} \end{pmatrix}
$$

这个乘积可以展开为：

$$
\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}} = \begin{pmatrix} d_1v_{11}^2 + d_2v_{21}^2 + d_3v_{31}^2 & d_1v_{11}v_{12} + d_2v_{21}v_{22} + d_3v_{31}v_{32} & d_1v_{11}v_{13} + d_2v_{21}v_{23} + d_3v_{31}v_{33} \\ d_1v_{12}v_{11} + d_2v_{22}v_{21} + d_3v_{32}v_{31} & d_1v_{12}^2 + d_2v_{22}^2 + d_3v_{32}^2 & d_1v_{12}v_{13} + d_2v_{22}v_{23} + d_3v_{32}v_{33} \\ d_1v_{13}v_{11} + d_2v_{23}v_{21} + d_3v_{33}v_{31} & d_1v_{13}v_{12} + d_2v_{23}v_{22} + d_3v_{33}v_{32} & d_1v_{13}^2 + d_2v_{23}^2 + d_3v_{33}^2 \end{pmatrix}
$$

#### 近似解释

在大系统假设下，当 (N) 非常大时，(\overline{\mathbf{V\}}) 中的元素 (v\_{ij}) 可以被认为是随机的并且独立同分布。根据大数定律，这些随机变量的平方和交叉项的平均值趋于某个固定值：

1. 对角线上元素 (d\_1v\_{11}^2 + d\_2v\_{21}^2 + d\_3v\_{31}^2) 等可以近似为 $$\frac{1}{N} \sum_{i=1}^{N} d_i$$，因为每个 (v\_{ij}^2) 的期望值大致为 $$\frac{1}{N}$$（这是由于正交矩阵的性质，即列向量是单位向量）。
2. 非对角元素 $$d_1v_{11}v_{12} + d_2v_{21}v_{22} + d_3v_{31}v_{32}$$ 的期望值由于独立性趋近于零，因为这些交叉项涉及不同的正交向量之间的乘积，其期望为零。

因此，整个矩阵可以近似为：

$$
\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}} \approx \frac{1}{N} \operatorname{tr}(\mathbf{D}_N) \mathbf{I}
$$

在我们的例子中，(\operatorname{tr}(\mathbf{D}\_N) = d\_1 + d\_2 + d\_3)，所以最终结果为：

$$
\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}} \approx \frac{1}{3} (d_1 + d_2 + d_3) \mathbf{I}
$$

这意味着在大系统情况下，复杂的矩阵乘积近似为对角矩阵，且对角线上所有元素均为 $$\frac{1}{3}$$ 的 (\mathbf{D}\_N) 的迹。这个近似大大简化了计算过程，并且在 (N) 足够大时是非常有效的。

#### 1. 正交矩阵的性质

正交矩阵 $$\overline{\mathbf{V}}$$ 的列向量是单位向量，并且这些列向量彼此正交。这意味着对于任意两个不同的列向量 $$\mathbf{v}_i$$ 和 $$\mathbf{v}_j$$（即 $$\overline{\mathbf{V}}$$ 的第 $$i$$ 列和第 $$j$$ 列）满足：

$$
\mathbf{v}_i \cdot \mathbf{v}_j = \sum_{k=1}^{N} v_{ki}v_{kj} = 0 \quad \text{(当 \(i \neq j\) 时)}
$$

其中 $$v_{ki}$$ 表示矩阵 $$\overline{\mathbf{V}}$$ 的第 $$k$$ 行第 $$i$$ 列的元素。

#### 2. 期望值的计算

假设 $$\overline{\mathbf{V}}$$ 的元素 $$v_{ki}$$ 是从某个分布中随机抽取的独立同分布随机变量。为了简化讨论，我们假设这些随机变量的期望值为 0，方差为 $$1/N$$（这是常见的 Haar 分布矩阵元素的性质）。

现在我们来看非对角元素，例如：

$$
d_1 v_{11}v_{12} + d_2 v_{21}v_{22} + d_3 v_{31}v_{32}
$$

在这个表达式中，每个乘积 $$v_{ki}v_{kj}$$（例如 $$v_{11}v_{12}$$）是来自正交矩阵不同列之间的乘积。

由于这些 $$v_{ki}$$ 是独立同分布的随机变量，且每个 $$v_{ki}$$ 的期望值为 0，所以它们的乘积的期望值也为 0：

$$
\mathbb{E}[v_{ki}v_{kj}] = \mathbb{E}[v_{ki}] \cdot \mathbb{E}[v_{kj}] = 0 \cdot 0 = 0 \quad \text{(当 \(i \neq j\) 时)}
$$

因此，整个非对角元素的期望值就是这些项的期望值的和，而每一项的期望值为 0，所以：

$$
\mathbb{E}[d_1 v_{11}v_{12} + d_2 v_{21}v_{22} + d_3 v_{31}v_{32}] = d_1 \mathbb{E}[v_{11}v_{12}] + d_2 \mathbb{E}[v_{21}v_{22}] + d_3 \mathbb{E}[v_{31}v_{32}] = 0
$$

#### 3. 大数定律的作用

即使在单次实现中这些非对角元素可能不是精确的零，但在 $$N$$ 非常大的情况下，根据大数定律，随机变量的平均值将趋于其期望值。因此，尽管个别乘积 $$v_{ki}v_{kj}$$ 可能是非零的，但是随着 $$N$$ 的增大，这些随机噪声会逐渐平均化，使得非对角元素的值趋近于零。

#### 总结

正是由于正交矩阵中不同列的向量之间的乘积在期望上为零，这导致了非对角元素的期望值也趋近于零。这也是为什么在大系统假设下，$$\overline{\mathbf{V}}^T \mathbf{D}_N \overline{\mathbf{V}}$$ 可以被近似为一个对角矩阵的原因，对角元素由 $$\mathbf{D}_N$$ 的迹给出。
