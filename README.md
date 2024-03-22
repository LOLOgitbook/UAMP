# Page

Avicii-Levels.srt

\


Abstract

Generalized Approximate Message Passing (GAMP) allows for Bayesian inference in linear models with non-identically independently distributed (n.i.i.d.) priors and n.i.i.d. measurements of the linear mixture outputs.

非独立同分布（_n.i.i.d._）解释

在统计分析中，_“_非独立同分布_”_（_n.i.i.d._）指的是一系列随机变量，这些变量不必遵循相同的概率分布，但彼此独立。这与独立同分布（_i.i.d._）变量相反，后者既独立又遵循相同的分布。_n.i.i.d._ 条件放宽了每个变量必须遵循相同分布的要求，允许统计模型中存在更广泛的变异性和复杂性。这一概念在处理多样化数据源和类型的领域（如机器学习和信号处理）中尤其相关。

It has been shown that the fixed points of GAMP correspond to the extrema of a large system limit of the Bethe Free Energy (LSL-BFE)

However, the convergence of (G)AMP can be problematic for certain measurement matrices.

we revisit the GAMP algorithm by applying a simplified version of the Alternating Direction Method of Multipliers (ADMM) to minimizing the LSLBFE.

We show convergence of the mean and variance subsystems in AMBGAMP and in the Gaussian case, convergence of mean and LSL variance to the Minimum Mean Squared Error (MMSE) quantities.

Introduction

\


\


Approximate Message Passing Update Rule

The Approximate Message Passing (AMP) algorithm employs iterative update rules to estimate the signals in systems modeled by linear measurements with added noise. The fundamental update rules for the AMP algorithm are as follows:

1. 1.Message Update: At every iteration $t$, for every variable node $i$ in the system, update the message as: m(t+1)i=f(∑jAijz(t)j+x(t)i)m\
   i\
   (\
   t\
   \+\
   1\
   )\
   \
   \
   \=\
   f\
   (\
   ∑\
   j\
   \
   A\
   i\
   j\
   \
   \
   z\
   j\
   (\
   t\
   )\
   \
   \
   \+\
   x\
   i\
   (\
   t\
   )\
   \
   \
   )\
   \
   \
   m\_i^{(t+1)} = f\left(\sum\_{j}A\_{ij}z\_j^{(t)} + x\_i^{(t)}\right)\
   mi(t+1) =f(∑j Aij zj(t) +xi(t) )\
   where $A\_{ij}$ represents the elements of the measurement matrix, $z\_j^{(t)}$ are the auxiliary variables, $x\_i^{(t)}$ the current estimate of the $i$-th variable, and $f(\cdot)$ a non-linear function derived from the prior distribution of the variables.\
   \
   \

2. \

3. 2.Auxiliary Variable Update: Simultaneously, update the auxiliary variable $z\_j^{(t)}$ based on the residual error and the Onsager correction term as: z(t+1)j=yj−∑iAijm(t+1)i+z(t)jN∑i∂f∂xi∣∣∣xi=x(t)iz\
   j\
   (\
   t\
   \+\
   1\
   )\
   \
   \
   \=\
   y\
   j\
   \
   −\
   ∑\
   i\
   \
   A\
   i\
   j\
   \
   \
   m\
   i\
   (\
   t\
   \+\
   1\
   )\
   \
   \
   \+\
   z\
   j\
   (\
   t\
   )\
   \
   \
   N\
   \
   ∑\
   i\
   \
   ∂\
   f\
   \
   ∂\
   x\
   i\
   \
   \
   \
   ∣\
   x\
   i\
   \
   \=\
   x\
   i\
   (\
   t\
   )\
   \
   \
   \
   \
   \
   z\_j^{(t+1)} = y\_j - \sum\_{i}A\_{ij}m\_i^{(t+1)} + \frac{z\_j^{(t)\}}{N}\sum\_{i}\frac{\partial f}{\partial x\_i}\bigg|\_{x\_i=x\_i^{(t)\}}\
   zj(t+1) =yj −∑i Aij mi(t+1) +Nzj(t) ∑i ∂xi ∂f ∣∣ xi =xi(t)\
   In this equation, $y\_j$ represents the $j$-th measurement, and $\frac{\partial f}{\partial x\_i}$ denotes the derivative of the update function with respect to its input, evaluated at the current estimate. $N$ is the total number of variable nodes.\
   \
   \

4. \


The AMP algorithm iterates these steps until a convergence criterion is met, such as a minimal difference in the estimate or reaching a maximal number of iterations. The convergence to meaningful estimates highly depends on the characteristics of the measurement matrix and the accuracy of the model assumptions regarding noise and signal priors

Onsager correction term

The Onsager correction term is calculated during each iteration of the AMP algorithm as part of the auxiliary variable update step. Specifically, it is given by:

\[ \text{Onsager correction term} = \frac{1}{N} \sum\_{i=1}^{N} \frac{\partial f(x\_i^{(t)})}{\partial x\_i} \cdot z\_j^{(t-1)} ]

where:

* \
  $N$ is the total number of variable nodes,\
  \
  \

* \

* \
  $\frac{\partial f(x\_i^{(t)})}{\partial x\_i}$ is the derivative of the non-linear function $f$ applied to the current estimate $x\_i^{(t)}$ of the $i$-th variable, and\
  \
  \

* \

* \
  $z\_j^{(t-1)}$ is the auxiliary variable from the previous iteration.\
  \
  \

* \


This term is subtracted from the update of the auxiliary variable to mitigate the effects of the correlations introduced by iterative updates, thereby enhancing the algorithm's stability and convergence.

\


\


\


\


\


\


\


\


\


\


\


\
