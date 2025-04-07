### Natural Language Generation(NLG)

review: text generation model

$$
p(y_t|\{y_{<t}\})=\frac{\exp{S_w}}{\sum_{w'}\exp{S_{w'}}}
$$

每一步都可以被看作是一个分类问题：将正确答案与其他部分分开

如何提升？

* 改进decode算法
* 改进training算法

> student forcing: 全部使用自己的数据，根据$y_1\cdots y_t$预测$y_{t+1}$
>
> teacher forcing: 使用ground truth：根据$\hat{y_1}\cdots\hat{y_t}$预测$y_{t+1}$

### Decoding

greedy:

$$
\hat{y_t}=\arg\max_{w\in V}P(y_t=w|y_{<t})
$$

beam search(in lec 7)

在close-ended tasks上效果好，open-ended效果并不好

self-amplification; 重复某个词后这个string的负最大似然值减小了

solutions:

* Hueristic: 不重复n-gram
* unlikelihood: 降低已经出现过的词的概率
* coverage loss：防止生成同样的词
* different decoding strategy：contrastive decoding，使用最大化logprob_LLM(x)-logprobSLM(x)的x

但是事实上人类说话的曲线并非是全接近1的，需要更好的decoding策略！：sampling！

**top-k：** 仅仅考虑概率最高的k个候选人，其余概率归为0. 

问题：无法处理动态概率分布！--如果概率分布比较平均，可能直接删除可能的候选人；如果概率分布集中在一个词上，仍然存在概率选择概率很低的词汇

**top-p：** 考虑何时从高到低的概率和累计到概率p，将余下的概率归0

**temperature：**

$$
P_t(y_t=w)=\frac{\exp(S_w/\tau)}{\sum_{w'\in V}\exp(S_{w'}/\tau)}
$$

Higher Temperature -- less diverse, 更集中在原先较大的词汇上

Lower Temperature -- more diverse, 分布更加平坦
**re-ranker:** 使用一个评分函数（比如评价生成的连贯性，风格等）对生成的候选人排序，选择最高的

注意可以混合多个re-ranker，来得到针对不同方面的评分函数

**Exposure Bias:** 模型训练的时候使用teacher-forcing，使用golden truth作为上文进行训练。但是输出的时候使用student-bias，使用自己的输出作为上文

* Scheduled Sampling: with some probability $p$, decode a token and feed as the next input, rather than 直接喂golden truth
* Dataset Aggregation(DAgger) 在某些时间段，从目前的模型中生成句子（可能有错误生成前缀，但是被正确后缀纠正过的），把这些句子加入模型作为数据集

### RLHF

Reward-Based, 希望模型可以学习人类的偏好，构建奖励函数

### Metrics

* vector similarity -- 计算句子embed之后的相似度
* word mover's distance --计算句子之间的距离
* BERT SCORE -- 使用预训练的BERT embedding，把候选的句子和reference setence使用cosine similarity匹配
* Sentence Movers Similarity: RNN embedding sentence + word mover's distance
* BLEURT: 基于BERT，衡量候选语句在语义上和原句子的相似度

open-ended

* MAUVE: 计算输出句子和ground truth的low-dimension概率表示，并使用KL divergence/move distance来计算表示的距离。low-dimension space可以有效提取语义

**Human Evaluations(Most Important)**

* slow and expensive
* inconsistent results
* not reproduceable
* not really logical

> ADEM: use human data to train a judge model
>
> HUSE: determine the human distribution with the output distribution
>
> Evaluate the model by interacting with them

**Other questions**

* 道德约束。但是仍然存在越狱方案
* Factual Errors
* LM learn harmful data from Internet training data

> Extensive Data Filtering -- 几乎不可能，太昂贵
>
> LM甚至有可能自己生成有害信息（即使不要求他这么做）
