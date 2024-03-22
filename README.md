# paper1

## Abstract

Generalized Approximate Message Passing (GAMP) allows for Bayesian inference in linear models with non-identically independently distributed (n.i.i.d.) priors and n.i.i.d. measurements of the linear mixture outputs.

_**非独立同分布（n.i.i.d.）解释**_

_在统计分析中，“非独立同分布”（n.i.i.d.）指的是一系列随机变量，这些变量不必遵循相同的概率分布，但彼此独立。这与独立同分布（i.i.d.）变量相反，后者既独立又遵循相同的分布。**n.i.i.d.** 条件放宽了每个变量必须遵循相同分布的要求，允许统计模型中存在更广泛的变异性和复杂性。这一概念在处理多样化数据源和类型的领域（如机器学习和信号处理）中尤其相关。_

It has been shown that the fixed points of GAMP correspond to the extrema of a large system limit of the Bethe Free Energy <mark style="color:red;">(LSL-BFE)</mark>

However, the convergence of (G)AMP can be problematic for certain measurement matrices.

we revisit the GAMP algorithm by applying a simplified version of the <mark style="color:red;">Alternating Direction Method of Multipliers (ADMM) t</mark>o minimizing the LSLBFE.

We show convergence of <mark style="color:red;">the mean and variance subsystems</mark> in AMBGAMP and <mark style="color:red;">in the Gaussian case</mark>, convergence of mean and LSL variance to the Minimum Mean Squared Error (MMSE) quantities.

## Introduction



#### Approximate Message Passing Update Rule

The Approximate Message Passing (AMP) algorithm employs iterative update rules to estimate the signals in systems modeled by linear measurements with added noise. The fundamental update rules for the AMP algorithm are as follows:

1. **Message Update**: At every iteration $t$, for every variable node $i$ in the system, update the message as: $$m_i^{(t+1)} = f\left(\sum_{j}A_{ij}z_j^{(t)} + x_i^{(t)}\right)$$ where $A\_{ij}$ represents the elements of the measurement matrix, $z\_j^{(t)}$ are the <mark style="color:red;">auxiliary variables</mark>, $x\_i^{(t)}$ the <mark style="color:red;">current estimate of the $i$-th variable,</mark> and $f(\cdot)$ a non-linear function derived from the prior distribution of the variables.
2. **Auxiliary Variable Update**: Simultaneously, update the auxiliary variable $z\_j^{(t)}$ based on the <mark style="color:red;">residual error</mark> and the <mark style="color:red;">Onsager correction</mark> term as: $$z_j^{(t+1)} = y_j - \sum_{i}A_{ij}m_i^{(t+1)} + \frac{z_j^{(t)}}{N}\sum_{i}\frac{\partial f}{\partial x_i}\bigg|_{x_i=x_i^{(t)}}$$ In this equation, $y\_j$ represents the $j$-th measurement, and $\frac{\partial f}{\partial x\_i}$ denotes the derivative of the update function with respect to its input, evaluated at the current estimate. $N$ is the total number of variable nodes.

The AMP algorithm iterates these steps until a convergence criterion is met, such as a minimal difference in the estimate or reaching a maximal number of iterations. The convergence to meaningful estimates highly depends on the characteristics of the measurement matrix and the accuracy of the model assumptions regarding noise and signal priors

&#x20;Onsager correction term

The Onsager correction term is calculated during each iteration of the AMP algorithm as part of the auxiliary variable update step. Specifically, it is given by:

\[ \text{Onsager correction term} = \frac{1}{N} \sum\_{i=1}^{N} \frac{\partial f(x\_i^{(t)})}{\partial x\_i} \cdot z\_j^{(t-1)} ]

where:

* $N$ is the total number of variable nodes,
* $\frac{\partial f(x\_i^{(t)})}{\partial x\_i}$ is the derivative of the non-linear function $f$ applied to the current estimate $x\_i^{(t)}$ of the $i$-th variable, and
* $z\_j^{(t-1)}$ is the auxiliary variable from the previous iteration.

This term is subtracted from the update of the auxiliary variable to mitigate the effects of the correlations introduced by iterative updates, thereby enhancing the algorithm's stability and convergence.

















