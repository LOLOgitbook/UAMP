# Generalized Approximate Message Passing for Estimation with Random Linear Mixing

## 公式123

在指数族分布中，概率分布通常可以表示为：

$$
p(x | \theta) = h(x) \exp(\eta(\theta)^\top T(x) - A(\theta))
$$

其中：

* x 是观测值。
* $$\theta$$  是参数。
* ( h(x) ) 是基础测度。
* ($$\eta(\theta)$$ ) 是自然参数。
* ( T(x) ) 是充分统计量。
* ( $$A(\theta)$$ ) 是对数配分函数（也即规范化常数的对数），确保分布归一化。

现在，我们考虑配分函数的对数 ( A(\theta) )。由于概率密度必须积分（或对于离散分布则是求和）到1，我们有：

$$
\exp(A(\theta)) = \int h(x) \exp(\eta(\theta)^\top T(x)) \, dx
$$

现在，如果我们对 ( $$A(\theta)$$ ) 关于参数$$\theta$$   进行微分，我们可以获得充分统计量的期望：

&#x20;

$$
\frac{\partial}{\partial \theta} A(\theta) = \frac{\partial}{\partial \theta} \log \left( \int h(x) \exp(\eta(\theta)^\top T(x)) \, dx \right)
$$

&#x20;

应用对数导数法则，我们得到：

&#x20;

$$
\frac{\partial}{\partial \theta} A(\theta) = \frac{1}{\exp(A(\theta))} \cdot \int h(x) T(x) \exp(\eta(\theta)^\top T(x)) dx
$$

&#x20;

因为 ( $$e^{-A(\theta)}$$ ) 就是 ($$p(x | \theta)$$的一部分，上式等同于：

&#x20; $$\frac{\partial}{\partial \theta} A(\theta) = \int T(x) p(x | \theta) \, dx$$ &#x20;

右边的积分表示的是充分统计量 ( T(x) ) 在参数 ( \theta ) 下的期望值，即 ($$\mathbb{E}[T(X) | \theta]$$

如果我们以 ( U ) 为随机变量，( V ) 为条件，和 ( \phi(U) ) 为充分统计量，那么我们可以有类似的推导。在这种情况下，配分函数 ( Z(V) ) 就变成了条件概率中的规范化常数，我们可以得到：

\[ $$\frac{\partial}{\partial v} \log Z(v) = \mathbb{E}$$\[U|V = v] ]

&#x20;\[ $$\frac{\partial^2}{\partial v^2} \log Z(v) = \frac{\partial}{\partial v} \mathbb{E}$$\[U|V = v] =  {Var}(U|V = v) ]]

其中，这个条件概率密度函数 ( p\_{U|V}(u|v) ) 与指数族的形式是相符合的，因为它表明了给定 ( V = v ) 时 ( U ) 的期望值和方差可以通过 ( \log Z(v) ) 来得到，这是通过对 ( \log Z(v) ) 求导数和求二阶导数来实现的。这也说明了为什么 ( T(x) ) 是一个充分统计量——因为它包含了分布的所有信息，这可以通过参数$$\theta$$  的函数( $$A(\theta)$$ ) 来完全确定。

在GAMP（Generalized Approximate Message Passing）算法中，之所以区分线性步骤（Linear Step）和非线性步骤（Nonlinear Step），主要是基于这些步骤处理数据的方式和所涉及的数学操作的性质。

#### # 线性步骤（Linear Step）&#x20;

这一步骤被称为“线性”是因为它涉及的主要操作是线性的。具体来说，它包括对估计值进行线性变换和线性组合。在GAMP算法中，线性步骤通常涉及以下操作：

* 使用矩阵 ( A ) 将估计的输入 ( \hat{x} ) 变换成中间变量 ( z )。这一变换是通过矩阵乘法完成的，矩阵乘法本质上是一种线性操作。
* 通过矩阵 ( A ) 的转置 ( A^T ) 和其他线性计算来更新和改进输入向量 ( \hat{x} ) 的估计。

线性操作保持加法和乘法的特性，即 ( A(bx + cy) = bAx + cAy )，其中 ( b ) 和 ( c ) 是常数，( x ) 和 ( y ) 是向量。这些操作不引入任何非线性元素，如指数、对数或其他更复杂的数学函数。

#### # 非线性步骤（Nonlinear Step）

非线性步骤则包括在数据处理中引入非线性函数，这些函数能够处理和建模变量间的非线性关系以及更复杂的分布特征。在GAMP算法中，非线性步骤涉及：

* 应用非线性的概率模型（如sigmoid函数或其他激活函数）来更新和细化估计值。
* 根据观测数据 ( y ) 和预测结果 ( \hat{z} ) 的差异，通过非线性函数调整残差，以改进估计准确性。

这些非线性函数可以是任何非线性的数学表达式，它们通常用于描绘输入和输出之间复杂的依赖关系，以及解决由于线性模型无法覆盖的数据特征的限制。

#### 总结

“线性步骤”和“非线性步骤”的命名直接反映了它们在数据处理中所采用的数学操作类型。线性步骤利用线性变换简化问题框架，为非线性步骤提供基础；而非线性步骤则通过引入非线性函数来增加模型的表达能力，使算法能够适应更广泛的应用场景和更复杂的数据结构。这种方法的组合优化了算法的性能和适用性，使其能够有效地解决各种估计问题。
