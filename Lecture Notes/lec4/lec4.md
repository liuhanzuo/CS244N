### Linguisitic Structure

dependency structure依赖结构： 例如找形容词在修饰谁。 是NLP重要的组成部分。 Why need it:需要合适的句子结构来表达复杂的意思（无法通过单个单词来传达）

歧义种类：

1. prepositional phrase attachment ambiguity
2. coordination scope ambiguity
3. adjectival/adverbial modifier ambiguity
4. verb phrase attachment

### Dependency Graph

对于一个句子，如何构造一个dependency graph： 每个单词使用箭头把它和dependent连接。

1. ROOT是根节点，只有一条出边
2. 整个图无圈

data source:使用人工标注，构造树库，优势在于：

1. highly reusable
2. broad coverage
3. frequencies and distributional information
4. evaluate NLP systems

**Projectivity：** 图中的箭头是否交叉，如果箭头不交叉，则称之为projectable

构造dependency parser的几种方法：

1. DP, 总共有$\frac{1}{n+1}\binom{2n}{n}$种总情况，但是存在$O(n^3)$的clever method
2. Graph Algorithm. Minimal Spanning Tree
3. Constraint Satisfaction
4. **Transition-based parsing/deterministic learning classfiers**

初始时$\sigma=[ROOT],\beta=w_1,\cdots,w_n,A=\emptyset$,这里$\sigma$表示即将进行操作的栈，$\beta$代表还未进行操作的单词，$A$代表已经建立的dependency

每一步可以选择三种操作之一：

1. shift: $\sigma,w_i|\beta,A\to\sigma|w_i,\beta,A$
2. left-arc: $\sigma|w_i|w_j.\beta,A\to\sigma|w_j,\beta,A\cup\{r(w_i,w_j)\}$
3. right-arc: $\sigma|w_i|w_j,\beta,A\to\sigma|w_i,\beta,A\cup\{r(w_i,w_j)\}$

使用神经网络来预测应该进行哪个操作。 NO/YES search（最简单的版本不需要search，但是可以使用search来提高performance）

**Evaluator**

使用自己的predictor预测，标记dependency和label，与tree bank中的结果比较

$$
UAS=\frac{\# \text{ correct dependencies}}{\# \text{ total dependencies}}\\
LAS=\frac{\# \text{ correct dependencies\& labels}}{\# \text{ total dependencies\&labels}}
$$
