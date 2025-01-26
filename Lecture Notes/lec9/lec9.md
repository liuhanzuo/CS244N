### BPE(byte-pair encoding)

NLP里面最小的单位是什么？单词？

单词难以处理新出现的词！（比如造词）

最新的模型可以学习单词的POS(pats of words,subword token)

但是，如何划分单词作为subword是关键

* start with a vocabulary containing only characters and an "end-of-word" symbol
* using a corpus of text, find the most common adjacent characters, add as another subword
* termiate until the subwords amount reach maximum

这样可以学习每一个常见subword的意思！

在处理新单词的时候，将他们分割成尽可能少的subword token！

pretrain有什么作用

* representations of language
* parameter initializations for strong models
* probability distributions over language we can sample

更像一个更好的"initialization"

pretrain的优势

* text data 非常多
* text data非常多样化，finetune后的结果相比直接finetune更好（可以理解语义）

### Model Pretraining Ways

**Encoders**

把某些单词替换为[MASK],根据上下文预测他们

$$
h_1,\cdots,h_T=Encoder(w_1,\cdots,w_T)\\
y_i\sim Ah_i+b
$$

$y_i$是输出，根据embedding matrix* representation+bias构成

$\tilde{x}$是masked的句子，实际上是在学习$p_\theta(x|\tilde{x})$

BERT: Bidirectional Encoder Representations from Transformers

为了加入“检测错误单词”之类的功能，BERT对选择mask的单词做如下操作：

* 15%的单词被masked
* masked单词80%被替换为[MASK]
* masked单词10%被替换为random token
* masked单词10% 保留原词（但是仍然预测它）

Interesting Idea: Segment Embedding 给一段单词标记为 $A$,另一段标记为$B$,让模型预测$B$是不是$A$的续写（长距离depedency能力）

```python
final embedding = token embedding + segment embedding + position embeddings
```

但是后来的工作证明segment embedding不是必要的（）

**Encoder-Decoder**

mask由自己的编号，decoder预测的target是输入mask的部分

优势：有双向预测能力，可以在machine translation一类的领域受欢迎

**Decoder**

$$
h_1,\cdots,h_T=Decoder(w_1,\cdots,w_T)\\
w_t\sim Ah_{t-1}+b
$$

### Finetuning

**Full Finetuning: adapt all parameters**

**Lighweight Finetuning: train a few existing/new parameters**

#### Prefix Tuning

Add a predix of parameters, freeze all pretrained parameters

用句子生成prefix parameters，再一起喂到LLM中

#### LoRA

$W\to W+AB$

$W\in R^{d\times d}$, $A\in R^{d\times k}$, $B\in R^{k\times d}$
