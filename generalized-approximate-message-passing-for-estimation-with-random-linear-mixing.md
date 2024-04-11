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
