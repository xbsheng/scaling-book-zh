---
layout: distill
title: "Training LLaMA 3 on TPUs"
# permalink: /main/
description: "Let's take a close look at how we'd train LLaMA 3 models on TPU v5p using what we've learned in the previous section. How big are they? How expensive is training in different configurations? How are they sharded? Let's work through some back-of-the-envelope estimates for how the previous sections map onto real models."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training"
previous_section_name: "Part 5: Training"

next_section_url: ../inference
next_section_name: "Part 7: Inference"

bibliography: main.bib

giscus_comments: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "What does LLaMA 3 look like?"
  - name: "Counting parameters and FLOPs"
  - name: "How to shard LLaMA 3-70B for training"
  - name: "Worked Problems"

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---_本节的目标是将前文的结果应用于一个非常实际的问题：训练LLaMA 3系列（herd）模型。与前几节不同，我们希望您能亲自完成大部分工作。因此，我们隐藏了每个部分的答案，以便您先尝试自己解答。不妨拿起笔亲手算一算！_

### LLaMA 3 长什么样？

LLaMA-3 模型家族<d-cite key="llama3"></d-cite> 包含三个主要模型：LLaMA 3 8B、70B 和 405B。我们将主要关注 70B 模型，并将 8B 和 405B 留给本章末尾的练习部分供您探索。以下是 LLaMA 3-70B 的架构，取自 LLaMA 的 [HuggingFace 页面](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json)。

| **超参数**                  | **值**    |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

为了突显查找这些信息的便捷性，这里展示了配置本身及其映射关系：

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

_为许多不同的开源大语言模型制作一张包含这些数字的大表格很有用，这样您就可以快速比较它们做出的设计决策。_

### 参数量与计算量（FLOPs）计算

**问题：** 根据上表，我们能计算出 LLaMA 3-70B 的参数量吗？🤫 让我们运用[第 4 节](../transformers)的内容，看看能否得出 70B！

| 参数类型         | 公式                                                                                                                                              | 计数                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW 参数         | d_model * d_ff * 3 (用于 SwiGLU 门控，以及上投影和下投影) * n_layers                                                                                         | 8,192 * 28,672 * 3 * 80 = **56.3e9**                         |
| 词表参数         | 2 (输入和输出嵌入) * n_embeddings * d_model                                                                                                       | 2 * 128,256 * 8,192 = **2.1e9**                              |
| 注意力参数       | n_layers * [ 2 (用于查询嵌入和拼接后的输出投影) * d_model * n_heads * d_qkv + 2 (用于键和值) * d_model * n_kv_heads * d_qkv]                      | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                           |

太棒了！我们得到了预期的数字。正如所料，您会注意到 FFW 参数在整体参数量中占据了绝对主导地位，尽管注意力部分的参数量也相当可观。

<p markdown=1 class="takeaway">**要点**：MLP 块中的三个大权重矩阵比 Transformer 中的所有其他数组都要大得多，以至于在分析模型内存或计算量时，我们通常几乎可以忽略所有其他参数。对于 LLaMA 3-70B，它们占据了 70B 参数中的 56B。</p>

现在我们来计算 FLOPs！*请记住[第 4 节](../transformers)中关于训练的一般规则。*

**问题：** LLaMA-3 在每个训练步骤中，每个 token 执行多少次浮点运算（FLOPs）？_这有助于我们确定整个训练过程的开销有多大。_

<details><summary>思考过后，请点击这里查看答案！</summary>


**答案**：如[第 4 节](../transformers)所示，每个 token 我们大约执行 $$6 \cdot \text{参数量}$$ 次浮点运算，因此这里大约是 `6 * 70e9 = 4.2e11` FLOPs / token。大约是每步每个 token 0.5 TFLOP。假设我们处于计算受限状态，在单个 TPU v5p 芯片上，假设完美利用浮点计算能力，这大约需要 `4.2e11 / 4.59E+14 = 1ms`。


</details>

**问题：** LLaMA 3 在大约 15 万亿个 token 上进行了训练。总计需要多少次浮点运算？

<details><summary>思考过后，请点击这里查看答案！</summary>


**答案**：这很简单，就是 `4.2e11 * 15e12 = 6.3e24 FLOPs`。6.3 尧浮点运算。这非常巨大！在单个 TPU 上，这将需要 `6.3e24 / 4.59E+14 = 435 年`。这也是很久！


</details>

**问题：** 假设我们想在一个拥有 16x20x28 = 8960 个芯片的完整 TPU v5p Pod 上进行训练。在 bfloat16 精度下，以 40% 的模型浮点计算利用率（MFU）进行训练需要多长时间（假设我们处于计算受限状态）？

<details><summary>思考过后，请点击这里查看答案！</summary>


**答案**：我们知道每个 TPU v5p 每秒可以执行 4.59e14 次浮点运算。在 40% MFU 下，这将花费大约 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6 秒`。**大约是 44 天！** 这相当合理，假设我们实际上能达到 40% 的 MFU。


</details>

**问题：** LLaMA 3-70B 的预训练使用了大约 400 万 token 的批大小（batch size）。我们至少需要多少个 TPU 才能以这个批大小进行训练？_您可以假设使用 bfloat16 格式的参数和 float32 格式的优化器状态，并且您每层对梯度进行 4 次检查点（checkpoint）保存。_

<details><summary>思考过后，请点击这里查看答案！</summary>


**答案**：这个问题主要涉及内存使用，因为这是对可用计算资源的唯一严格约束。在训练期间，我们对高带宽内存（HBM）有三个主要用途：模型参数、优化器状态和梯度检查点。如果我们假设权重为 bfloat16，优化器状态为 float32，并采用一种_非常_保守的梯度检查点方案（每层 4 次），则有：

| **参数**         | 2 * 70GB        | ~140GB  |
| **优化器状态**   | 8 * 70GB        | ~560GB  |
| **梯度检查点**   | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **总计**         |                 | ~21.6TB |

此处的总计约为 21.6TB。您会注意到，即使采用非常保守的检查点方案，梯度检查点也在内存占用中占绝对主导地位。从技术上讲，我们可以改为每层 1 个检查点，或者进行微批处理（microbatching），但这是一个合理的估算。根据这些假设，由于每个 TPU v5p 拥有 96GB HBM，我们需要 `21.6e12 / 96e9 = 225` 个 TPU。实际上这并不多！

*为什么我们不这么做呢？* 嗯，因为这将需要我们 `44 天 * 8960 / 225 = 1752 天` 来完成训练。这将近四年。**时间太长了。** 尽管如此，这也清楚地表明，我们使用这些大型集群并非因为内存受限，而是因为我们需要额外的计算能力（FLOPs）。


</details>

**问题：** 在与上题相同的假设下，如果我们使用 8960 个 TPU v5p 芯片，每个芯片将使用多少内存？

<details><summary>思考过后，请点击这里查看答案！</summary>


**答案**：我们的总内存仍然是约 21.6TB，因此每个芯片我们大约使用 2.4GB，这基本上算不了什么。如果我们采用更激进的检查点策略，例如每层 12 个检查点，我们也只会达到每个芯片 8GB。在如此规模的训练中，我们远未达到内存受限的状态。


</details>

<p markdown=1 class="takeaway">**要点**：在非常小的拓扑结构上训练甚至非常大的模型在技术上是可行的，但需要注意的是训练时间可能会很长。能够计算一次训练运行的总 FLOPs 使我们可以通过假设一个适中的 MFU 和一个已知的拓扑结构来估算其训练时间。</p>

### 如何分片 LLaMA 3-70B 进行训练

让我们延续上面的设定，假设我们想在拥有 8960 个芯片的 TPU v5p Pod 上，以 400 万 token 的批大小（每批 1024 个长度为 4096 的序列）训练 LLaMA 3-70B。让我们讨论一下针对此模型的最佳分片（sharding）策略。

**问题：** 根据上述假设，我们能否仅使用完全分片数据并行（FSDP）来训练我们的模型？首先，假设我们无法进行任何序列/上下文并行（sequence/context parallelism）。_这应该是您想到的第一个方法，因为它简单，如果可行的话不会引入额外的通信开销。_

<details><summary>思考过后，请点击这里查看答案！</summary>


**答案**：这个答案会有点学究气。如前所述，LLaMA 3-70B 最初是用长度为 4K 的序列训练的，因此 400 万 token 的批大小给了我们一个*序列批大小（sequence batch size）* 为 1024。这意味着我们最多只能进行到 1024 路的纯数据并行/FSDP，_因为那就是我们用于数据并行的序列数量_。所以，从“无额外通信的完全数据并行”这个简单意义上讲，答案是否定的。下一个问题将回答一个稍微不那么学究气的版本。


</details>

**问题：** 让我们放宽不进行任何序列分片的要求。如果我们允许自己对批和序列两个维度都进行 FSDP，那么能否在 8960 个芯片上仅使用 FSDP 训练 LLaMA 3-70B？

<details><summary>思考过后，请点击这里查看答案！</summary>


**答案**：现在我们允许自己进行序列/上下文并行，我们的扩展性就大得多了。首先计算每个设备的批大小。如果我们进行 8960 路 FSDP，每个 TPU 的批大小将是 `4 * 1024 * 1024 / 8960 = 468` 个 token。我们从前一节知道，当 $$\text{每个设备的批大小} < 2550 / M_X$$ 时，我们会受到 FSDP 引入的 ICI 通信限制。由于我们可以使用完整的 3D Pod 专门化 3 个轴，这将给我们一个下限 850，而我们远低于此值。**所以答案是否定的，即使使用 3 个轴也不行。我们肯定会受限于通信。**


</details>

**问题：** 现在让我们看看混合张量并行（tensor parallelism）和 FSDP。是否存在某种组合能让我们保持计算受限状态？如果存在，我们应该进行多少 FSDP 和多少张量并行？

<details><summary>思考过后，请点击这里查看答案！</summary>


**答案**：首先检查一下这样是否可行。我们知道，如果每个芯片的批大小小于 $2550^2 / 2F = 113$，我们将受限于通信。正如我们在上面看到的，我们略高于此值。这很好！现在要选择最优的 FSDP 数量，我们可以使用公式

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

四舍五入到一个合理的 2 的倍数，这大约给我们 2048 路 FSDP 和 4 路张量并行。这应该能很好地工作！


</details>

<p markdown=1 class="takeaway">**要点**：我们可以使用数据并行（1024 路）、序列并行（2 路）和张量并行（4 路）的组合，在完整的 TPU v5p Pod 上以 400 万 token 的批大小训练 LLaMA-3，而不会受限于通信。如果我们尝试进行纯 FSDP 或 FSDP + 序列并行，则会受限于通信。我们在上一节中推导出的方程非常实用。</p>
## Worked Problems
**问题 1 [将 LLaMA 70B 扩展到更多芯片]：** 假设我们希望在 4 个算力单元上以相同的批大小(batch size)训练 LLaMA 3-70B。我们会采用什么并行方案？计算会受限于计算瓶颈(compute bound)还是通信瓶颈(communication bound)？训练大约需要多长时间？*请确保使用正确的屋顶线模型限制。*

**问题 2 [LLaMA 405B]：**

(a) 使用 LLaMA 3-405B 的[配置文件](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)，参照上文格式整理一个包含所有关键超参数的表格。该模型的总参数量是多少？每个训练步骤的浮点运算数(FLOPs)是多少？若训练 15T 个 token，总共需要执行多少次浮点运算？

(b) 假设我们计划在 8 个 TPU v5p 算力单元上进行训练。我们会采用什么并行方案？训练需要多长时间？计算会受限于计算瓶颈还是通信瓶颈？

<h3 markdown=1 class="next-section">第六节内容到此结束。关于第七节 Transformer 推理，请点击[这里](../inference)。</h3>