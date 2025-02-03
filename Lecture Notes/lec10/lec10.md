### Guest Lecture--Multitask Assistant

如果我们希望LLM可以很好的预测下一个句子，那么事实上它应该学会一些人类的背景知识

### Zero-shot and Few-shot ICL

GPT2-LLM可以进行zero-shot learning（回答在数据集中没有出现过的问题）

GPT3-LLM可以进行few-shot learning

GPT3 在大数加法方面比较弱，需要改变prompt（CoT）

### Instruction Finetuning

构造一些prompt和回答，在上面finetune

* Positive: straightforward and can generalize to unseen tasks
* Negative: Comments are expensive for human source data
* open-ended question are hard
* language modeling penalizes all token-level mistakes equally, but some errors are worse than others
* Mismatich between LM objects and human preference

### RLHF

$R(s)\in\mathbb{R}$是一个奖励函数

最大化

$$
\mathbb{E}_{\tilde{s}\sim p_\theta(s)}[R(\tilde{s})]

$$

$$
\theta_{t+1}:=\theta_t+\alpha\nabla_{\theta_t}\mathbb{E}_{\tilde{s}\sim p_\theta(s)}[R(\tilde{s})]
$$

如何估计期望项？

$$
\nabla_\theta\mathbb{E}_{\tilde{s}\sim p_\theta(s)}[R(\tilde{s})]=\sum_s R(s)\nabla_\theta p_\theta(s)=\sum_s R(s)p_\theta(s)\nabla_\theta\log p_\theta(s)\\
=\mathbb{E}_{\tilde{s}\sim p_\theta(s)}[R(\tilde{s})\nabla_\theta\log p_\theta(\tilde{s})]\approx\frac{1}{m}\sum_{i=1}^mR(s_i)\nabla_\theta \log p_\theta(s_i)
$$

“强化好的行为，增加概率，让他们在未来更有可能出现；减少坏的行为，进小概率，让他们在未来不太可能出现“

$$
R(s)=RM_\phi(s)-\beta\log\left(\frac{p_\theta^{RL}(s)}{p_\theta^{PT}(s)}\right)
$$

不希望finetune过后的model和原模型偏离许多

InstructGPT: 把RLHF的任务变多，成百上千个

* step1:collect demonstration data, and train a supervised policy（感觉就是instruction tuning）
* step2: collect comparison data and train a reward model
* step3: optimize a policy against the reward model using reinforcement learning

ad&dis

* directly model preferences,generalize bryonf labeled data
* RL is very tricky to get right

### Limitations of RL+Reward

* 人类偏好非常不可靠--subjective
* 很难仅仅使用一个数字来衡量人类的偏好
* 可能捏造事实/产生幻觉
* 用人类偏好训练的模型更加不稳定！

**目前的所有算法都是data-expensive的！需要大量的人工标注以达到比较好的效果**

AI feedback -- AI generate, use AI evaluate its response!
