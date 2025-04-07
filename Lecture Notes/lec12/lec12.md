### Question Anwsering

关注"unstructured text"

#### Reading Comprehension

阅读理解，提问一个文章中涉及到的信息

同时，阅读理解任务也是其他任务的基础：例如Information Extraction(信息提取)，Semantic Role Labeling(语义人物标记)

数据集：SQuAD， 文章来自wikipedia，100~150个单词左右；问题的答案由很少的单词构成（但是并非所有任务都可以使用一两个此回答，这是弊端）

每个问题有几个答案，所有答案的平均作为最终答案

评分分为exact match(0/1) 和 F1(partial credit)

$$
\text{final score} = \max\{\text{exact match score}\}+\max\{F1\text{ score}\}
$$

#### LSTM model

**LSTM Layer**

from seq2seq model with attention to this LSTM

BiDAF: the bidirectional attention flow model

用GloVE作为word embedding

$$
e(c_i)=f\left([GloVe(c_i);charEmb(c_i)]\right),e(q_i)=f([GloVe(q_i);charEmb(q_i))
$$

bidirectional LSTMs 去分别产生contextual embeddings for both context and query

$$
\overrightarrow{c_i}=LSTM\left(\overrightarrow{c_{i-1}},e(c_i)\right)\in R^H\\
\overleftarrow{c_i}=LSTM\left(\overleftarrow{c_{i-1}},e(c_i)\right)\\
c_i=[\overleftarrow{c_i},\overrightarrow{c_i}]
$$

$$
\overrightarrow{q_i}=LSTM\left(\overrightarrow{q_{i-1}},e(q_i)\right)\in R^H\\
\overleftarrow{q_i}=LSTM\left(\overleftarrow{q_{i-1}},e(q_i)\right)\\
q_i=[\overleftarrow{q_i},\overrightarrow{q_i}]
$$

**Attention Layer**

保存注意力，用于阅读理解

context-to-query attention:选择与query word语义最相近的context words

* compute a similarity score for every parit of $(c_i,q_j)$
* $$
  S_{i,j}=w_{\text{sim}}^T[c_i;q_j;c_i\cdot q_j]\in R,w_\text{sim}\in R^{6H}
  $$
* context-to-query attention
* $$
  \alpha_{i,j}=softmax_j(S_{i,j})\in R,a_i=\sum_{j=1}^M \alpha_{i,j}q_j\in R^{2H}
  $$
* quey-to-context attention
* $$
  \beta_i=softmax_i(\max_{j=1}^M(S_{i,j}))\in R^N, b_i=\sum_{j=1}^N \beta_jc_j\in R^{2H}
  $$

  $$
  \text{final output: }g_i=[c_i,a_i;c_i\cdot a_i;c_i\cdot b_i]\in R^{8H}
  $$

**Modeling Layer**

* pass $g_i$ to another two layers of bi-directional LSTMs
* sttnetion layer is modeling interations between query and context
* Modeling layer is modeling interations within context words

  $$
   m_i=BiLSTM(g_i)\in R^{2H}
  $$

**Output Layer**

use two claddifiers predicting the start and end positions $p_{start}$ and $p_{end}$

$$
p_{start}=softmax(w_{start}^T[g_i;m_i),p_{end}=softmax(w_{end}^T[g_i;m_i'),m_i'=BiLSTM(m_i)\in R^{2H},w_{start},w_{end}\in R^{10H}
$$

**Training Loss**

$$
L=-\log p_{start}(s^*)-\log p_{end}(e^*)
$$

#### BERT Model

使用一个pretrain-BERT model， 用BERT encoder, $h_i$是hidden vector of $c_i$

$$
p_{start}(i)=softmax_i(w_{start}^T h_i), p_{end}(i)=softmax_i(w_{end}^T h_i)
$$

任务是去预测在reference text中开始和结束位置

#### Comparison

* BiDAF: attention between context and query(c,q)+(q,c)
* BERT: attention betweem all: (c,c), (c,q), (q,c), (q,q)
* work better if add a (c,c) attention on BiDAF

#### SpanBERT

* masking continuous span of words instead of 15% random masking!
* use two end points of span to predict all the masked words in between -- compress all the information of the span to the endpoints(use two endpoints to predict the words between)

证明了可以设计更好的pretrain目标来达到更好的效果.

但是阅读理解类的问题并没有解决：如果在文段末尾插入了一句与目标完全无关的话，表现的结果将会答复下降。此外，如果在数据集1上finetune的BERTmodel没有办法在数据集2上得到类似的结果，表现能力降低很多：无法generalize

### Open -domain QA

不给定文章，在大数据集下提取问题

#### retriever-reader framework

检索（retriever）：找到相关文章

阅读（reader）：根据检索的文章回答问题

$$
\text{Input: }D=\{D_1,D_2,\cdots,D_N\}\\
\text{Output: }anwser string A
$$

**retriever could also be trained**

使用(question, anwser)的pair去训练  

coastal QA model: pretrained model(T5)+finetune on QA dataset

也可以使用最近邻搜索，不使用reader， encode所有段落

### QAs

* Q：为什么某一个数据集上模型效果好，但是换到另一个数据集模型无法很好的generalize
  A：每个数据集都有一些表层规律，比如词语之间的重叠，模型善于捕捉这些规律，从而到另一个数据集上的时候会因为失去这些规律导致表现下滑
* Q:如果更换数据集，比如使用google上真实用户的提问和回答，是否还会出现之前数据集无法generalize的问题
  A:是的，也会出现。这是因为即使没有之前的表层规律问题，人们询问的问题大多相似。回答类似的问题的能力不应当被认为是有泛化能力的
* Q:如果答案位置和检索到的文章距离过远，是否还呢个正确得到答案（因为使用的是最近邻搜索，如果相距很远未必可以得到答案）？
  A：并不知道，不保证可以得到答案，但是实际结果上来看，没有什么问题。
* Q:回答open-domain的问题是否需要常识？
  A:是的，即使是现在（2021），常识也是一个很难解决的问题
