### Distributed Representations

使用一个$d$维的向量来作为word embedding， 要求相似的词汇有相似的vector embedding

POS，dependency label也被写作一个 $d$维向量

对于所有可能的待选项，我们使用word+POS+dependency来标记他们（作为embedding）

### Deep Learning Classifiers are non-linear classifiers

$$
p(y|x)=\frac{\exp (W_y,x)}{\sum_{c=1}^C\exp(W_c,x)}
$$

训练矩阵 $W\in R^{C\times d}$使用：

$$
loss=\sum_i-\log p(y_i|x_i)
$$

输入的向量是由concat不同部分的向量得来的（比如查看buffer中的第一个词和stack中的后两个次，concat）

如何制作一个更精准的model？

* 更大，更深的神经网络
* 搜索技巧--beam search
* CRF(conditional random field)

graph-based dependency parser: 设计一个"nueral sequence model"

效果更好！但是复杂度不是$O(n)$而是$O(n^2)$

Basic techniques: regularization, droupout, activation layer, parameter initialization(使用小的随机数初始化，否则没有梯度.e.g.$Uniform(-r,r)$Xavier initialization: $Var(W_i)=\frac{2}{n_\text{in}+n_\text{out}}$)

learning rate: muanually modify/exponential decay

### n-gram LM

语言模型（language model）意味着我们预测$x_{t+1}$ given $x_1\cdots x_t$

n-gram:只考虑前面的$n$个单词，其余的忽略，即

$$
P(x_{t+1}|x_1,\cdots x_t)=P(x_{t+1})(\text{Markov Assumption})\\
P(x_{t+1}|x_1,\cdots,x_t)=P(x_{t+1}|x_{t-n+1}\cdots x_t)(n-\text{gram Assumption})
$$

概率从哪里获得：数据集中出现的次数，即在$x_1\cdots x_t$之后出现$x_{t+1}$的次数/出现$x_1\cdots x_t$的次数

问题：如果句子长度很长，那么有可能出现sparsity problem: 1.$w$从来没在数据集中出现过--那么$w$出现的概率为0 2.$x_1,\cdots x_t$在数据集中从未出现过，概率无法计算！（其实可以不用管）

complexity：$O(\#\text{total words}^n)$

Generation:根据已有的参数计算条件概率

Result：生成出来的话并不连贯（make no sense）

### Window Based Nueral Model

将一个固定长度的前缀放入神经网络，embedding，concat，然后计算下一个单词的概率，softmax sampling

Problem:没有解决window长度的问题

Advantage：1.可以distribution处理数据（embedding）$\to$语义相近的词有相近的embedding $\to$在下一个词有相似的概率

2.没有sparsity problems

3.不需要存储所有的n-gram

### RNN

$x_i$生产出output$y_i$,并提供hidden state$h_i$给$x_{i+1}$.

hidden state updating:

$$
h^t=\sigma\left(W_hh^{t-1}+W_ee^t+b_1\right)
$$

word embedding:

$$
e^t=Ex^t,x^t\text{ is a one-hot vector}\in R^{|V|}
$$

output distribution:

$$
\hat{y^t}=softmax\left(Uh^t+b_2\right)\in R^{|V|}
$$

Advantages;

1. 可以处理任意长度的输入
2. 理论上可以使用历史上的信息
3. model size不随输入长度增长
4. 对称性

Disadvantages:

1. 无法并行，计算速度慢
2. 如果需要的信息距离当前时间点很远，很难得到（经过了很多层，仍然保留的历史信息其实不多）
