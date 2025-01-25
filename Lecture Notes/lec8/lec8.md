### Attention-based Networks

attention可以注意到很远的单词，不需要像RNN一样$O(n)$的时间

*attention is like a lookup table*

lookup table：给定query，依次比对keys，output对应的value

attention：soft-match,每一个keys有一个对应的weight，输出 $\sum weight\times value$

**Math Part:**

$w_{1:n}$表示一个句子，由词汇表$V$中的词汇构成

对于每一个query $w_i$, $x_i=Ew_i$,$E\in R^{d\times |V|}$be the embedding matrix

$$
\text{queires }q_i=Qx_i,\text{keys }k_i=Kx_i, \text{values }v_i=Vx_i
$$

$$
e_{i,j}=q_i^T k_j,\alpha_{i,j}=\frac{\exp(e_{i,j})}{\sum_{j'}\exp(e_{i,j'})}
$$

$$
o_i=\sum_j\alpha_{i,j}v_j
$$

$o_i$是输出向量，代表了"average value"

$e_{i,j}=x_i^TQ^TKx_i$,那么为什么需要两个矩阵$Q$,$K$? 事实上，最终学习的是$Q^TK$ 的low-rank approximation，用于简化计算！

### Building Blocks For Attention

|                      Barriers                      |                  Solutions                  |
| :-------------------------------------------------: | :------------------------------------------: |
|      Doesn't have an inherent notion of order      |  Add position representations to the inputs  |
|  No nonlinearities for DL! just a weighted average  | aplly same FFN to each self-attention output |
| Need to ensure do not look at future for prediction |                                              |

**1.sequence order**

把每个句子中单词的位置序号表示为一个向量positional embedding!

$$
p_i\in R^d,i\in\{1,2\cdots,n\}
$$

真正的embeidding：

$$
\tilde{x_i}=x_i+p_i,x_i\text{is calculated from self-attention block}
$$

* sinusoidal position representations: concatenate sinusoidal functions of varying periods]
* $$
  p_i=\begin{matrix}
  \sin\left(10000^{2\times 1/d}\right)\\
  \cos\left(10000^{2\times 1/d}\right)\\
  \cdot\\
  \cdot\\
  \cdot\\
  \sin\left(10000^{2\times \frac{d}{2}/d}\right)\\
  \cos\left(10000^{2\times \frac{d}{2}/d}\right)\\
  \end{matrix}
  $$
* absolute position representations:$p_i$ be learnable parameters: learn a matrix $p\in R^{d\times n}$
* variations: relative linear position attention, dependency syntax-based position

**2. Nonlinearilties**

Easy fix: add a FFN to post-process each output vector

$$
m_i=MLP(output_i):=W_2\times ReLU\left(W_1\times output_i+b_1\right)+b_2
$$

**3.No future information**

Mask Attention for decoder

$$
e_{i,j}=\begin{cases}
q_i^T k_j, j\le i\\
-\infty, j>i\\
\end{cases}
$$

### Transformers

Intuition:使用一个head来平均所有attention比较困难，希望使用多个head来表示不同特征

$$
X=[x_1,\cdots,x_n]\in R^{n\times d}\text{ be the concatenation of input vectors}
$$

如同上面所说的，我们需要计算的attention是$XQK^TX^T$,softmax,然后与XV相乘

Multi-head attention:

$$
Q_l,K_l,V_l\in R^{d\times \frac{d}{h}}
$$

这里$h$ 是attention head的个数

$$
output_l=softmax\left(XQ_lK_l^TX^T\right)\times XV_l,output_l\in R^{\frac{d}{h}}
$$

$$
output=[output_1,\cdots,output_h]\times Y,Y\in R^{d\times d}
$$

Reality: reshape!

* 计算$XQ\in R^{n\times d}$, reshape to $R^{n\times h\times\frac{d}{h}}$
* transpose to $R^{h\times n\times \frac{d}{h}}$

如同把$XQ$按column切片为$h$份，把$K^TX^T$按行切片分为$h$份

**Scale dot product**

当维数比较大的时候，dot product的结果会比较大，不能很好反应softmax，所以更改公式为

$$
output_i=softmax\left(\frac{XQ_lK_l^TX^T}{\sqrt\frac{d}{h}}\right)\times XV_l
$$

**Residual Connection**

$$
x^i=x^{i-1}+Layer(x^{i-1})
$$

**Layer Norm**

帮助模型训练加速！

每个layer中normalize单位方差和标准差

 $x\in R^d$是word vector,$\mu=\frac{1}{d}\sum_{j=1}^dx_j$ is mean, $\sigma=\sqrt{\frac{1}{d}\sum_{j=1}^d\left(x_j-\mu\right)^2}$ is standard deviation

$\gamma\in R^d$, $\beta\in R^d$ 是可以学习的"gain","bias" 参数

$$
output=\frac{x-\mu}{\sqrt \sigma+\epsilon}\times\gamma+\beta
$$

每个词的layernorm都是独立的，并不共享

**Decoder Block**

* masked self-attention
* Add&Norm
* FF
* Add&Norm

**Encoder Block**

* self-attention
* Add&Norm
* FF
* Add&Norm

**Encoder-Decoder!**

多了一个步骤：cross-attention

$h_1\cdots h_n$是Transformer encoder的输入，$z_1,\cdots,z_n$是Transformer decoder的输入

keys,values从encoder中来,queries 从decoder中来

$$
k_i=Kh_i,v_i=Vh_i,q_i=Qz_i
$$

注意到 这里decoder的输入是生成结果，即首先输入$<SOS>$(开始符)，接下来按照encoder的输出预测下一个词，作为输入，反复生成直到结束

### Questions With Transformer

平方增长的GPU内存量，

总量$O(n^2d)$
