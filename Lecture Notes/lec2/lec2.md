### Skip Gram with Negative Sampling

$$
J_t(\theta)=\log \sigma(u_o^T v_c)+\sum_{i=1}^k \mathbb{E}_{j\sim p(w)}\left[\log\sigma(-u_j^Tv_c)\right]
$$

在计算上，可以写作：

$$
J_\text{neg}=-\log \sigma(u_o^T v_c)-\sum_{k\in\text{sample}}\log\sigma(-u_k^T v_c)
$$

这里面$P(w)=U(w)^{\frac{3}{4}}/Z$, 其中$U(w)$是初始的概率分布，$Z$是归一化系数。

Advantages:使概率分布更加均匀，缩小出现次数较少的词汇与出现次数多的词汇之间的差别

### Co-occurence Matrix

使用SVD计算，保留奇异值较大的那些行和列，可以使用较小的空间保留更加完整的特征

log-bilinear model: $w_i\cdot w_j=\log P(i|j)$, 方便计算vector differences: $w_x(w_a-w_b)=\log\frac{P(x|a)}{P(x|b)}$

GLoVe: 使用单一化的loss function

$$
J=\sum_{i,j=1}^Vf(X_{i,j})\left(w_i^T\tilde{w_j}+b_i+\tilde{b_j}-\log X_{i,j}\right)^2
$$

Intuition:忽略bias项$b_i$,$b_j$,希望$w_i^T\tilde{w_j}$与$\log X_{i,j}$更加接近

|                 Intrinsic                 |                  Extrinsic                  |
| :---------------------------------------: | :-----------------------------------------: |
| Evaluation on a specifc/intermediate task |          Evalation on a real task          |
|              Fast to Compute              |         Take a long time to compute         |
|      Helps to understand the system      | unclear for subsystem's problem/interaction |
|      Not helpful without a real task      |  A better subsystem, a better performance  |

word2vec测试： $a:b::c:?$, 希望找到？对应的词汇：

$$
d=\arg\min_i\frac{\left(x_a-x_b+x_c\right)^Tx_i}{\|x_a-x_b+x_c\|}
$$

如何应对多义词？两种解决方案

1. 1 word multiple vectors. 实战中不太实用；每个不同的意思单独作为一个向量，重复训练，直到每个词汇属于不同的聚类。问题：训练繁琐；”算作不同词义“的界限本身就比较模糊
2. 1 word 1 vector. $v_x=\sum\alpha_iV_{x_i},\alpha_i=\frac{f_i}{\sum_j f_j}=\mathbb{E}_\alpha[v_x]$

External Reading:[Linear Algebraic](./Linear Algebraic Structure of Word Senses, with Applications to Polysemy.pdf)
