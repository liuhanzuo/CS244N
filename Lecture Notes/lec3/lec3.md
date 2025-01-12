### Name Entity Recognition(NEG)

Recognition name/location in a sentence.

### Backpropogation

输入：$x$,经过一些变换: $z=Wx+b,h=f(z)$, $s=u^Th$, $J_t(\theta)=\sigma(s)=\frac{1}{1+\exp{-s}}$

一下偏导等式成立：

$$
\begin{cases}
	\frac{\partial}{\partial u}(u^T h)=h^T\\
	\frac{\partial}{\partial z}(f(z))=diag(f'(z))\\
	\frac{\partial}{\partial b}(Wx+b)=I
\end{cases}
$$

注意到在多层神经网络的时候，对底层参数的计算需要使用**链式法则**，可以通过梯度的复用来实现。

shape convention:导数的形状与原参数的形状保持一致，方便进行梯度下降（计算梯度时可能需要加上转置（transpose））

每个节点（变量）中存有upstream gradient, local gradient, downstream gradient， 可能存在多条连出/连入的情况。

例子：$h=f(z)$, 这个变换作为一个节点，存储的upstream gradient是$\frac{\partial s}{\partial h}$, local gradient是$\frac{\partial z}{\partial h}$, downstream gradient是$\frac{\partial s}{\partial z}=\frac{\partial s}{\partial h}\cdot\frac{\partial h}{\partial z}$,如果存在多个upstream gradient,就把他们与对应的local gradient相乘再相加，即为关于目标变量的upstream gradient

Manual Gradient Check:可以使用定义：

$$
f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}\approx\frac{f(x+h)-f(x)}{h}_{h=1e-4}
$$

来进行估计
