# Variance Predictions in VAMP/UAMP with Right Rotationally Invariant Measurement Matrices for niidGLM



## lemma 1

这段描述是关于一种数学定理（Lemma 1），用于处理特定类型的矩阵和它们的性质。这个定理主要关注在大规模随机矩阵的背景下，矩阵函数的平均行为。&#x20;

定理的主要内容是：

给定条件：

* (P) 是一个有界谱范数的厄米矩阵。这意味着(P)的所有特征值的绝对值都不会超过某个固定的实数上限。
* (V \in \mathbb{R}^{N \times M}) 是一个从Haar分布（一种在单位圆上均匀分布的随机矩阵分布）中选取的M列的子矩阵，这里M小于N。
* (B) 是一个非负定的矩阵，其谱范数（即最大特征值的绝对值）有一个上限。
* (D) 是一个只包含正项的对角矩阵。

定理表述： 当你构造一个由这些矩阵(V, P, B, D)构成的特定矩阵函数，并计算它的平均行踪（迹），那么这个平均行踪会趋近于一个简化形式的行踪的值。具体来说，就是 \[ \frac{1}{N} \text{tr} \left\[ B\left(VPV^T + D\right)^{-1} \right] - \frac{1}{N} \text{tr} \left\[ B(eI + D)^{-1} \right] \rightarrow 0 ] 几乎肯定地趋于0。这意味着这两个行踪的值在大N的极限下是相等的。

这里的标量(e)是通过以下方程组得到的唯一解（固定点）： \[ e = \frac{1}{N} \text{tr} \left\[ P \left(eP + (1 - e)eI\right)^{-1} \right], \quad e = \frac{1}{N} \text{tr} \left\[ B(eI + D)^{-1} \right] ]

简而言之，这个定理告诉我们，在处理具有特定分布（如Haar分布）的随机矩阵时，某些复杂矩阵表达式的平均行为可以通过一个更简单的表达式来近似。这个结果在理论和实际应用中非常有用，因为它为分析和计算提供了便利，尤其是在处理大规模数据或复杂系统时。





