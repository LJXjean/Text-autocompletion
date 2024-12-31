# 模型评估报告 (Evaluation Report)

本报告基于前述文本自动补全 (Text Autocompletion) 系统，在进行了一轮评估 (Evaluation) 后，得到了以下自动化指标 (Automatic Evaluation Metrics) 与示例预测结果。

---

## 1. 评估背景 (Background)

在文本自动补全任务中，我们为模型提供一段已有的 **context**（上下文），让模型生成后续文本 (Continuation)。常见的评价指标有：

1. **BLEU (Bilingual Evaluation Understudy)**
2. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
3. **Perplexity (PPL)**
4. **BERTScore**

在本次评测中，我们主要报告了 **BLEU** 和 **ROUGE** 分数。若需要进一步分析模型的语言流畅度及语义一致性，亦可结合 **Perplexity** 和 **BERTScore**。

---

## 2. 数据与流程 (Data & Procedure)

- **评测数据 (Test Set)**：从已有的 (Context, Reference Continuation) 对中选取 50 条作为测试集。
- **评测流程 (Evaluation Procedure)**：
  1. 对每条测试样本，使用模型在给定 `context` 的位置生成文本 (Prediction)；
  2. 与“真实后续” (Reference) 进行对比；
  3. 计算以下指标：
     - **BLEU**：基于词或 n-gram 的重叠程度衡量预测与参考文本的相似度。
     - **ROUGE**：评估预测文本和参考文本在字词或句子片段的重叠度，常见变体包括 ROUGE-1、ROUGE-L 等。
     - **Perplexity** (可选)：衡量模型对文本分布的困惑度，越低表示模型对该文本更“自信”。
     - **BERTScore** (可选)：通过深度语言模型对预测与参考的隐向量表征进行相似度计算，较好地衡量语义一致性。

---

## 3. 评测结果 (Evaluation Results)

从日志中可见：
```
   ============ EVALUATION RESULTS ============
Number of samples evaluated: 50
Average BLEU:    0.0008
Average ROUGE-1: 0.1442
Average ROUGE-L: 0.0942
============================================

Sample predictions:

--- Example 1 ---
Context:    Twin brothers from south London have made a name for themselves on social media by cleaning things t...
Reference:  an a British Airways Airbus A320 at Heathrow Airport.

After almost two hours of cleaning, the twins...
Prediction: ave into the world of street cleaners.

"I was surprised when I saw the Instagram picture of the air...

--- Example 2 ---
Context:    The firm in charge of selling The Body Shop in the UK has set a deadline for buyers to submit bids i...
Reference:   after Aurelius, a private equity firm, put the company into administration in February only a few m...
Prediction: .

In a statement, a spokesperson said: "We have agreed to meet the deadline, which means that The B...
```

## 5. 综合分析 (Analysis)

1. **意料之中的低分**  
 - 目前为了做一个demo演示项目许多技术都是尽可能的从简（没有微调），并且没有使用很强的大语言模型，所以效果自然不会很好。如果关注“语义一致性”和“信息准确度”，还可使用 **BERTScore** 或进行主观评测 (Human Evaluation) 以作进一步分析。

2. **后续改进**  
 - **增强检索 (RAG)**：提升检索文档数量与质量，使模型生成内容更贴近上下文事实；  
 - **Fine-Tuning**：对新闻领域文本进行更深入的训练或使用 LoRA/PEFT 进行领域微调；  
 - **Prompt Engineering**：改进 Prompt 模板，把检索到的信息与用户上下文有机结合；  
 - **使用更强大的 LLM**：如 LLaMA 2、Falcon 等，在更丰富数据上预训练，往往在文本理解和生成方面具有更强表现。

---

## 6. 其他可能指标 (Other Metrics)

- **Perplexity (PPL)**：  
- 反映模型在预测参考文本时的自信度。若 perplexity 值较高，则说明模型并没有很好地“匹配”该文本的语言分布。  
- **BERTScore**：  
- 基于深度语言模型（如 BERT、RoBERTa）的向量表示，对预测和参考进行逐词或逐 Token 的语义相似度比较，比纯粹的 n-gram 相似更能衡量真实的语义重合度。
