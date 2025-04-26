# LAMP

1\.

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









