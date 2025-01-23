### RNN

$$
h^t=\sigma\left(W_hh^{t-1}+W_xx^t+b_1\right)
$$

训练过程：

$$
loss=XE(y^t,\hat{y^t})
$$

$\hat{y^t}$是一个one-hot vector(给定前t-1个单词，预测第t个，真实的单词概率为1，其余为0)，希望预测的概率分布和真实的概率分布比较接近

反向传播：注意由chain rule

$$
\nabla W_h=\sum_{t=1}^T\nabla W_h^t
$$

即将$T$步的梯度相加作为$W_h$的梯度。注意到这需要大量的串行计算，比较慢，有时也进行truncated backpropogation：只考虑前$k$个layer的梯度，以此作为标准对$W_h$进行更新，可以加快训练过程。这被称为teacher-forcing

生成：从每一步的概率分布中sample一个（softmax）

### Vanishing/Exploding Gradient

RNN的梯度是会消失/爆炸的

以$\sigma(x)=x$举例，

$$
\frac{\partial h^t}{\partial h^{t-1}}=diag\left(\sigma'(W_hh^{t-1}+W_xx^t+b_1)\right)W_h=W_h
$$

$$
\frac{\partial J^i(\theta)}{\partial h^j}=\frac{\partial J^i(\theta)}{\partial h^i}\prod_{j<t\le i}\frac{\partial h^t}{\partial h^{t-1}}=\frac{\partial J^i(\theta)}{\partial h^i}\times W_h^{i-j}
$$

考虑$W_h$的分解（特征值，特征向量），如果它的特征值都小于1，那么当$i-j$足够大的时候梯度必然趋于0；如果他的某一个特征值大于1，那么当$i-j$足够大的时候梯度必然趋于$\infty$

梯度爆炸的处理方案：Gradient Clipping

$$
\theta\to\theta-lr\times\nabla\theta\\
\theta\to\begin{cases}
\theta-lr\times\nabla\theta \text{ if}\|\nabla\theta\|\le\text{threshold}\\
\theta-lr\times\text{threshold}\times\frac{\nabla\theta}{\|\nabla\theta\|}\text{ otherwise}
\end{cases}
$$

### LSTM

long short-term memory RNNs

在时间点$t$,我们有一个hidden state$h^t$和一个cell state $c^t$,均为长为$n$的向量

* cell存储长期信息
* LSTM可以向cell中read，store，erase，write信息
* gates是一些长为$n$的向量，每个时间点，gate可以为open(1) or close(0) or somewhere between, 三个gate控制哪些信息被erased/written/read

input: $x^t$, hidden state $h^t$, cell state $c^t$

$$
f^t=\sigma\left(W_fh^{t-1}+U_f x^t+b_f\right)\\
i^t=\sigma\left(W_ih^{t-1}+U_i x^t+b_i\right)\\
o^t=\sigma\left(W_oh^{t-1}+U_o x^t+b_o\right)
$$

* 遗忘门控制从前一个cell中保留/遗忘哪些信息
* 输入门控制哪些cell中的新信息被写入到cell
* 输出门控制哪些cell中的信息被输出到hidden中

$$
\tilde{c^t}=\tanh\left(W_ch^{t-1}+U_cx^t+b_c\right)\\
c^t=f^t\times c^{t-1}+i^t\tilde{c^t}\\
h^t=o^t\times\tanh c^t 
$$

$\tilde{c^t}$是新的待写到cell state里面的内容

不会像RNN那样梯度爆炸（但是有可能其他方面导致梯度爆炸）

梯度爆炸/消失并不是针对于RNN模型的问题，而是共有的问题（底层梯度流苏很慢）

### Bidirectional RNN

$$
h_1^t=RNN_{FW}\left(h_1^{t-1},x^t\right)\\
h_2^t=RNN_{FW}\left(h_2^{t+1},x^t\right)\\
h^t=[h_1^t,h_2^t]
$$

上面的RNN也可以是LSTM
