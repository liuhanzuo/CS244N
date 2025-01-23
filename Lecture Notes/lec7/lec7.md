### Machine Translation

**Statistical Machine Translation SMT**

给定一种语言句子$x$,找另一种$y$,使得

$$
y=\arg\max_x P(y|x)=\arg\max_x P(x|y)P(y)
$$

第一项$P(x|y)$代表translation model, 第二项$P(y)$代表fluency model

使用latent variable（对齐向量）， 改为考虑$P(x,a|y)$

某些语言可能无法对齐，比如"the". many-to-one, one-to-many, many-to-many

**Independent Assumption：** 可以把句子分解，翻译为若干个独立成分

1990s-2013，使用非常复杂的模型

* seperately-designed subcomponent
* feature engineering
* extra resources
* human effort

2014 Nueral Machine Translation NMT

$$
x\to^{\text{encode}}_{NN1}vector\to^{\text{decode}}_{NN2}y
$$

对于decoder的概率分布，每一个位置计算一个loss，最终loss为所有loss的平均值

**Multilayer-RNN**

更多层数的LSTM

2-4 layers best for encoder RNN , 4 layer best for decoder RNN

beam search：decode的时候每次允许保留k个候选人，对这k个人每个人生成k个可能的后续，仍然从中选取最可能的k个

问题：如果句子较长，那么会有显著更低的分数（但是可能是合理的翻译）

解决方案：normalization, 使用$\frac{1}{t}\sum_{i=1}^t\log P(x_i|x_1,\cdots,x_{i-1})$

缺点：

* less intepretable
* difficult to control

BLEU:使用human-wriiten translation进行比较

similarity code: n-gram+penalty for too short

BLEU是实用的 但是不完美--翻译方式有很多种，BLEU分数的高低并不能完全代表翻译水平

NMT的问题：

* out-of-vocabulary words
* domain mismatch
* low resource language
* pronoun resolution error
* morphological agreement errors

RNN的问题：information bottleneck, RNN系列的encoder都是使用一个状态来表示整个句子，传递的信息只能都被存储在其中，十分有限（可能丢失重点）
