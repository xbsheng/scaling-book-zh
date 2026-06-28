---
layout: distill
title: "All About Transformer Inference"
# permalink: /main/
description: "Performing inference on a Transformer can be very different from training. Partly this is because inference adds a new factor to consider: latency. In this section, we will go all the way from sampling a single new token from a model to efficiently scaling a large Transformer across many slices of accelerators as part of an inference engine."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training"
previous_section_name: "Part 6: Training LLaMA"

next_section_url: ../applied-inference
next_section_name: "Part 8: Serving LLaMA"

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
  - name: "The Basics of Transformer Inference"
  - subsections:
    - name: "What do we actually want to optimize?"
    - name: "Linear operations: what bottlenecks us?"
    - name: "What about attention?"
    - name: "Theoretical estimates for LLM latency and throughput"
    - name: "What about memory?"
    - name: "Modeling throughput and latency for LLaMA 2-13B"
  - name: "Tricks for Improving Generation Throughput and Latency"
  - name: "Distributing Inference Over Multiple Accelerators"
  - subsections:
    - name: "Prefill"
    - name: "Generation"
    - name: "Sharding the KV cache"
  - name: "Designing an Effective Inference Engine"
  - subsections:
    - name: "Continuous batching"
    - name: "Prefix caching"
    - name: "Let's look at an implementation: JetStream"
  - name: "Worked Problems"
  - name: "Appendix"

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
---

## The Basics of Transformer Inference
你已经训练好了一个Transformer模型，并希望用它来生成新序列。_说到底，基准分数上涨和损失曲线下降，只是衡量当模型真正投入使用时是否会产生有趣结果的代理指标！_<d-footnote>从历史上看，你可以进行大量不涉及推理的Transformer研究——基于评分的多项选择题基准测试可以在没有适当KV缓存或生成循环实现的情况下高效运行。这意味着，尤其是在研究代码库中，推理代码路径上常常有很多唾手可得的优化机会。</d-footnote>

在概念上，采样过程很简单。我们将一个序列输入，我们心爱的Transformer会输出$$\log p(\text{next token}_i \vert \text{previous tokens})$$，即所有可能下一个token的对数概率。我们可以从这个分布中采样以获得一个新token。将这个token追加到序列末尾并重复此过程，我们就得到了作为提示词延续的token序列。

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>图：</b> Transformer的朴素采样。蓝色的logits给出了我们可以采样的下一个token的分布。请注意，每一步都会重新处理整个前缀，导致该算法的运行时间为 $\Theta(n^2)$。" %}

我们刚刚描述的是Transformer采样的朴素实现，虽然它能工作，**但我们绝不会在实践中这样做**，因为每次生成一个token时，我们都在重新处理整个序列。这个算法在FFW部分的计算复杂度为$$O(n^2)$$，在注意力机制部分的计算复杂度为$$O(n^3)$$，才能生成$$n$$个token！

**我们如何避免这种情况？** 事实证明，我们可以不必每次都执行完整的前向传播。相反，我们可以从每次前向传播中保存一些中间激活值，从而避免重新处理之前的token。具体来说，由于在点积注意力中，一个给定的token只关注之前的token，我们可以简单地将每个token的键(key)和值(value)投影写入一个名为**KV缓存(KV cache)**的新数据结构。一旦我们保存了这些过去token的键/值投影，未来的token就可以直接计算它们的$$q_i \cdot k_j$$乘积，而无需对之前的token进行任何新的浮点运算。这很巧妙！

有鉴于此，推理有两个关键部分：

* <b style="color: red;">预填充 (Prefill)</b>：给定一个长提示词，我们同时处理提示词中的所有token，并将生成的激活值（具体来说，是键-值投影）保存在一个**"KV缓存"**中。我们还会保存最后一个token的logits。
* <b style="color: blue;">生成 (Generation)</b>：给定一个KV缓存和上一个logits，我们从logits中增量采样一个token，将该token反馈给Transformer，并为下一步产生一组新的logits。我们还将新token的KV激活值追加到KV缓存中。我们重复此过程，直到遇到特殊的`<EOS>` token或达到某个最大长度限制。

以下是使用KV缓存进行采样的示意图：

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>图：</b> 使用KV缓存的高效Transformer采样示意图。<b style=\"color: red;\">预填充</b> 处理我们的提示词，并将所有逐token的键-值激活值保存在缓存中。<b style=\"color: blue;\">生成</b> 接收此缓存（以及最后一个token的logits），采样一个新token，并将该新token通过模型传递，同时关注KV缓存并将其键-值投影保存回缓存。这是一个在MLP块中为 $O(n)$ 的算法。" %}

通过使用KV缓存进行采样，我们将生成$n$个token的时间复杂度在FFW部分降低到了$$O(n)$$，在注意力部分降低到了$$O(n^2)$$，因为我们不再重新处理之前的token。然而，生成一个序列仍然需要许多次前向传播——当你查询Gemini或ChatGPT并且结果流式返回给你时，发生的就是这种情况。每个token通常都是一个（部分缓存的）独立的、针对巨大模型的Transformer调用。

我们很快会看到，<b style="color: red;">预填充</b>和<b style="color: blue;">生成</b>是非常不同的任务——Transformer推理实际上是披着伪装的两项任务！与训练相比，KV缓存也是一个新颖且重要的复杂性来源。

### 我们到底想要优化什么？

在进一步深入之前，值得强调推理的一个全新方面：延迟。在训练期间，我们只关心吞吐量（**每芯片**每秒处理的token总数），而在推理期间，我们必须关注生成token的速度（包括**首token延迟 (Time To First Token, TTFT)** 和**逐token延迟**）。例如：

* 用于评估和数据生成的**离线批量推理**只关心推理的总体成本，而对单个样本的延迟不敏感。
* **聊天接口/流式任务**需要大规模廉价运行，同时要有低TTFT，并且生成token的速度要快于人类阅读速度。
* **边缘推理 (Edge inference)**（例如在你的笔记本电脑上运行`llama.cpp`）只需要以最低可能延迟服务单个用户，且可能面临严苛的硬件约束。

最大化硬件利用率仍然至关重要，有助于控制成本和TTFT，但与训练不同，它并不*必然*在所有场景下转化为更好的用户体验。在加速器、系统和模型架构层面的许多优化，都需要在延迟、吞吐量、上下文长度甚至模型质量之间进行权衡。

### 更细粒度的Transformer视图

到目前为止，我们主要将Transformer视为一系列前馈块的堆叠。虽然从浮点运算次数和内存的角度来看这通常是合理的，但它不足以正确建模推理。<d-footnote>你会注意到贯穿本节的一个现象是，推理比训练要“苛刻”得多。我们通常拥有的浮点运算次数要少得多，批处理的机会更少，并且对延迟的敏感性要高得多。KV缓存也使推理变得更加复杂。</d-footnote>正如我们在[第4部分](../transformers)中看到的，Transformer前向传播的主要组成部分是：

1.  **一系列线性运算**，包括MLP（$W_{in}$, $W_{out}$）以及注意力机制的QKV投影和输出投影（$W_Q$, $W_K$, $W_V$, $W_O$）。这些运算都涉及从HBM读取参数和一批激活值，进行一些浮点运算，然后将结果写回HBM。
2.  **点积注意力**。我们需要从HBM读取一批键-值投影和一批查询激活值，进行几次内积运算和一些softmax操作，然后将注意力结果写回HBM。
3.  **其他所有操作**，包括应用层归一化、激活函数、token采样、更新KV缓存以及位置编码。这些确实需要一些浮点运算，但它们被上述运算所主导，或融合到其中。

在接下来的几个部分，我们将在预填充和生成的背景下逐一审视这些操作，并探讨什么可能会成为性能的瓶颈。在单个加速器内部，我们是计算受限还是内存受限？我们要强调预填充和生成的答案会有多么不同。

### 线性运算：什么限制了我们？

所有的线性运算在概念上都是相同的，无论它们位于MLP块还是注意力块中。它们的算术强度取决于批大小。我们在[第1节](../roofline)中做过这个数学计算，但值得重复一遍。让我们看一个形状为$\text{bf16[B, D]}$的批次乘以一个形状为$\text{bf16[D, F]}$的矩阵的单次矩阵乘法。这可能是大的MLP块（$W_\text{in}$或$W_\text{out}$）或较小的注意力投影之一（$W_Q$, $W_K$, $W_V$, $W_O$）。要进行这个矩阵乘法，我们需要将这两个数组从HBM加载到MXU，执行乘法，然后将结果写回HBM。如前所述，我们有：

$$T_\text{math} = \frac{\text{计算浮点运算次数}}{\text{加速器浮点运算次数/秒}} = \frac{2BDF}{\text{加速器浮点运算次数/秒}}$$

$$T_\text{comms} = \frac{\text{通信字节数}}{\text{带宽 字节/秒}} = \frac{2BD + 2FD + 2BF}{\text{带宽 字节/秒}}$$

TPU或GPU可以通过在计算的同时加载数据来重叠这两个过程，因此要成为计算受限的，我们需要$$T_\text{math} \geq T_\text{comms}$$，即：

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{加速器浮点运算次数/秒}}{\text{带宽 字节/秒}} \underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} = 240$$

其中右侧是我们硬件的算术强度。现在，假设$D$和$F$相对于$B$非常大（通常我们的批次大小最多为500，而$D$和$F > 10k$），我们可以利用$\small{2BD + 2DF + 2BF \approx 2DF}$的事实来简化分母，得到：

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + 2BF} \approx \frac{2BDF}{2DF} \geq \frac{\text{加速器浮点运算次数/秒}}{\text{带宽 字节/秒}} \\
\underset{\text{TPU v5e}}{=} \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

如果我们量化权重或为矩阵乘法使用较低精度的浮点运算，这个临界批大小会发生变化。例如，如果我们将权重量化为int8或fp8，$B_\text{crit}$会减半。如果我们使用int8或fp8进行浮点运算，$B_\text{crit}$会翻倍。因此，如果我们令 $\beta = \text{每参数比特数} / \text{每激活值比特数}$ 且 $\alpha_\text{hbm} = C / W_\text{hbm}$，那么我们的临界批大小实际上是 $B_\text{crit} = \beta \alpha_\text{hbm}$。

<p markdown=1 class="takeaway">**要点：** Transformer矩阵乘法是计算受限的*当且仅当*每个副本的**token**批大小大于 $B_\text{crit} = C / W_\text{hbm} \cdot (\text{每参数比特数} / \text{每激活值比特数}) = \beta \cdot \alpha_\text{hbm}$。在TPU v5e上使用bf16激活值时，这个值为240个token。对于H100，约为280个token。</p>

在训练期间，由于我们跨一个非常大的批次复用相同的权重，我们所有的矩阵乘法都会具有很高的算术强度。**这种高算术强度也延续到了预填充阶段，因为用户提示词通常长达数百甚至数千个token。** 正如我们之前看到的，TPUv5e的硬件算术强度为240，因此如果一个超过240个token的序列被输入到运行在此硬件上、使用bf16的密集模型中，我们预期它是计算受限的，一切正常。短于此长度的提示词理论上可以批处理在一起以实现更高的利用率，但这通常不是必须的。

<p markdown=1 class="takeaway">**要点：** 在预填充期间，所有矩阵乘法基本上总是计算受限的。因此，简单地最大化硬件利用率或MFU（模型浮点运算利用率）足以最大化每芯片吞吐量（成本）和延迟（以TTFT的形式）。除非提示词非常短，否则在提示词级别进行批处理只会增加延迟，对预填充吞吐量的改善很小。</p>

然而，在生成期间，对于每个请求，我们一次只能处理一个token，因为步骤之间存在顺序依赖性！因此，我们只能（容易地）通过将多个请求批处理在一起、沿批维度并行化来实现良好的利用率。我们稍后会详细讨论这一点，但实际上，将许多并发请求批处理在一起而不影响延迟是很困难的。因此，**在生成期间，要充分利用硬件的浮点运算能力要困难得多。**

<p markdown=1 class="takeaway">**要点：** 在生成期间，总token批大小必须大于$B_{\text{crit}}$才能使线性/前馈操作成为计算受限的（在TPU v5e上使用bf16参数时为240）。因为生成是逐token串行发生的，这要求我们将多个请求批处理在一起，而这很难做到！</p>

*值得注意的是这个数字有多大！* 生成批大小为240意味着同时有240个并发请求在生成，并且对于密集模型有240个独立的KV缓存。这意味着这在实践中很难实现，除了在一些批量推理场景中。相比之下，在预填充期间处理超过240个token是相当常规的，尽管随着稀疏性的增加需要一些谨慎处理。

**请注意，这个具体数字会因量化方式和硬件类型而异。** 加速器通常能以较低精度提供更多算力。例如，如果我们有int8参数但以bf16进行计算，临界批大小会降至120。如果激活值和参数都是int8，它会跳回240，因为TPUv5e能提供400 TOPs/s的int8 x int8算力。

### 那注意力机制呢？

当我们看点积注意力操作时，情况变得更加复杂，特别是因为我们必须考虑KV缓存。让我们只看一个使用纯多头注意力机制的注意力头。在单次Flash Attention融合中，我们<d-footnote>我们在这里做了相当程度的简化，忽略了在应用softmax、掩码等操作中的非矩阵乘法浮点运算。这些操作应该与计算或HBM读取重叠，但在某些TPU世代上实现这一点可能并非易事。虽然这些细节不改变主要信息（即KV缓存通常是内存带宽受限的），但值得注意。</d-footnote>：

1.  从HBM读取形状为$\text{bf16[B, T, D]}$的$Q$激活值。
2.  从HBM读取$KV$缓存，这是一对形状为$\text{bf16[B, S, D]}$的张量。
3.  在$$QK$$矩阵乘法中执行$2BSTD$次浮点运算。使用Flash Attention，我们不需要将$\text{bf16[B, S, T]}$注意力矩阵写回HBM。
4.  在注意力$$AV$$矩阵乘法中执行$2BSTD$次浮点运算。
5.  将生成的$\text{bf16[B, T, D]}$张量写回HBM。

将所有部分综合起来，我们得到：

$$\text{多头注意力算术强度} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

对于预填充，$S=T$，因为我们进行的是自注意力(self-attention)，所以这简化为 $T^2 / 2T = T / 2$。这很好，因为这意味着**预填充期间注意力机制的算术强度是 $\Theta(T)$**。这意味着要成为计算受限的注意力机制相当容易。只要我们的序列长度相当大，就没问题！

但由于生成阶段的序列维度是1，且$B$和$D$维度相互抵消，我们可以作近似：

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

这很糟糕，因为这意味着在生成期间我们无法采取任何措施来提高注意力机制的算术强度。我们在加载巨大的KV缓存的同时只进行了少量的浮点运算。**所以，在注意力机制部分，我们基本上总是内存带宽受限的！**

<p markdown=1 class="takeaway">**要点：** 在预填充期间，对于任何合理的序列长度（大约 $\gt 480$ 个token），注意力机制通常是计算受限的；而在生成期间，我们的算术强度很低且恒定，因此总是内存带宽受限。</p>

*从概念上讲，为什么会这样？* 主要是因为在模型的线性部分，参数（内存带宽密集型组件）被许多批次项复用，所以我们是计算受限的。然而，每个批次项
## Tricks for Improving Generation Throughput and Latency
自原始[《Attention is All You Need》论文](https://arxiv.org/abs/1706.03762)发表以来，已开发出多种提升模型效率的技术，其中许多专门针对KV缓存（KV cache）。总体而言，较小的KV缓存更利于提升生成阶段的批量大小（batch size）和上下文长度（context length）而不损害延迟性能，并能简化Transformer周边系统（如请求缓存）的运作。若暂不考虑质量影响，我们可以观察到以下技术：

**分组多查询注意力（Grouped multi-query attention，又称GMQA、GQA）：** 我们可以在注意力机制中减少KV头（KV heads）的数量，并与多个查询头（Q heads）共享。极端情况下，所有查询头可共享单个KV头。相比纯多头注意力（MHA），这能将KV缓存缩减为查询头与KV头比例的倒数，且已观察到模型性能对此变化相对不敏感。

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

这也有效提升了注意力计算的算术强度（arithmetic intensity）（参见[第4节](../transformers)的问题4）。

**混合部分局部注意力层（Local attention layers）：** 局部注意力将上下文限制在中等规模的最大长度内。在训练和预填充阶段，这需要将注意力矩阵掩码为对角线带状区域而非三角形。这实质上限制了局部层KV缓存的最大长度。通过在模型中将局部层与全局层混合，当上下文超过局部窗口时，KV缓存尺寸将大幅缩减。

**跨层共享KV（Sharing KVs across layers）：** 模型可学习以某种模式跨层共享相同的KV缓存。虽然这能减小KV缓存尺寸，并在提升批量大小、缓存优化、离线存储等方面带来益处，但共享的KV缓存可能需要多次从高带宽内存（HBM）读取，*因此未必能提升单步推理时间*。

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="
 <b>左图：</b>纯全局注意力的多层结构。<b>右图：</b>全局/局部层交错并与相邻层共享的示例。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai技术博客</a>。"%}

**量化（Quantization）：** 推理过程通常对参数和KV缓存的精度敏感性较低。通过对参数和KV缓存进行量化（例如转为int8、int4、`fp8`等格式），我们可以节省两者的内存带宽，降低达到计算天花板所需的批量大小，并通过更大批量运行来节省内存。量化技术的优势在于：即使模型未经量化训练，通常也能在训练后应用。

**使用非对齐HBM读取与分页注意力（Paged Attention）：** 前述计算中为每个KV缓存分配了8k上下文，但实际常无需从内存读取完整KV缓存——请求长度分布广泛且通常不会用尽模型的最大上下文长度，因此我们可实现仅读取KV缓存非填充部分的计算核（如Flash Attention变体）。

分页注意力<d-cite key="paged"></d-cite>在此基础上进一步优化，将KV缓存存储在操作系统风格的页表中，基本避免了KV缓存的填充操作。虽然增加了复杂度，但确保每个批次仅使用必需的内存。这属于运行时优化，因此同样与具体架构无关。

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>图示：</b>生成阶段，单个标记（\"forth\"）访问多个KV缓存块/页。通过分页机制，我们避免加载或存储超出需求的内存。图片源自<a href=\"https://arxiv.org/pdf/2309.06180\">分页注意力论文</a>。" %}

<p markdown=1 class="takeaway">**宏观视角：** 总体而言，这些KV缓存优化技术可将KV缓存尺寸降低一个数量级以上（相较于标准MHA Transformer），从而可能使Transformer的整体成本降低一个数量级。</p>
## Distributing Inference Over Multiple Accelerators
到目前为止，我们一直略过了如何扩展到单个芯片之外的问题。遵循[第5节](../training)的方法，让我们探讨可供选择的不同策略及其权衡。一如既往，我们将分别考察预填充（prefill）和生成（generation）阶段。

### 预填充

从屋顶线模型（roofline）的角度来看，**预填充几乎与训练相同**，几乎所有相同的技术和权衡都适用——模型并行（Megatron parallelism）、序列分片（用于足够长的上下文）、流水线（pipelining），甚至完全分片数据并行（FSDP）都是可行的！你只需要保留KV缓存（KVs）以便稍后进行生成。与训练中一样，增加芯片数量可以为我们提供更多的FLOPs/秒（可能降低首次输出延迟TTFT），但也会增加通信开销（可能降低每芯片的吞吐量）。

**预填充分片的一般规则：** 这里是一套关于预填充的通用规则。我们假设仅对单个序列进行预填充（无批处理维度）：

1. *模型分片（Model sharding）：* 我们通常首先进行一定量的模型并行，直到受到ICI通信的限制。正如我们在[第5节](../training)中所看到的，对于1个轴，这大约是 $F / 2200$（通常约为4-8路分片）。
2. *序列并行（Sequence parallelism）：* 除此之外，我们进行序列并行（类似于数据并行，但在序列维度上进行分片）。虽然序列并行在注意力计算中引入了一些额外的通信，但在较长的上下文中，这通常相当小。与训练中一样，我们可以重叠通信和计算（分别使用集体矩阵乘法用于Megatron和环形注意力用于序列并行）。

<p markdown=1 class="takeaway">**要点：** 在预填充期间，几乎所有在训练期间有效的分片方式都可以正常工作。进行模型并行直到ICI约束，然后进行序列并行。</p>

### 生成

生成比预填充更为复杂。一方面，获得较大的批处理大小（batch size）更困难，因为我们需要将许多请求组合在一起。延迟目标更低。这些共同意味着我们通常更加受限于内存带宽（memory-bound），并对通信开销更敏感，这限制了我们的分片策略：

1. **FSDP不可行：** 由于在将参数和KV缓存从HBM加载到MXU时我们受内存带宽限制，我们不希望通过慢几个数量级的ICI来移动它们。*我们希望移动激活值（activations）而不是权重（weights）*。这意味着类似于FSDP的方法通常完全不适用于生成。<d-footnote>训练后意外地将其保留，是导致性能下降一个数量级的一个常见且简单的方式</d-footnote>

2. **没有理由进行数据并行：** 纯数据并行没有帮助，因为它复制了我们的参数，并不能帮助我们更快地加载参数。你最好启动模型的多个副本。<d-footnote>我们指的是，启动多个服务器，每个服务器包含模型的副本，并使用较小的批处理大小。在模型层面的数据并行严格来说更差。</d-footnote>

3. **没有序列就没有序列分片。** 祝你好运尝试序列分片。

_这主要留给我们模型分片（model sharding）的变体用于稠密模型生成_。与预填充一样，我们能做的最简单的事情是简单的模型并行（激活值完全复制，权重在MLP的隐藏维度上完全分片），直到我们受到ICI限制的4-8路。然而，由于我们通常受内存带宽限制，实际上我们可以超过这个限制来改善延迟！

**关于生成中ICI约束的说明：** 在训练期间，我们希望受计算限制，因此我们的屋顶线模型关注ICI通信时间超过FLOPs时间的情况。然而，在生成期间，如果我们因参数加载而受内存带宽限制，我们可以增加模型分片超过此点，并以最小的吞吐量代价（以每芯片每秒token数计）改善延迟。更多的模型分片为我们提供了更多的HBM来加载权重，而我们的FLOPs无关紧要。<d-footnote>意思是FLOPs时间没有成为瓶颈，所以我们需要担心的是ICI时间超过参数加载时间。</d-footnote> 让我们看看在ICI通信成为瓶颈之前，我们可以进行多少路模型并行。

$$\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

其中 $\beta = W_\text{hbm} / W_\text{ici}$。对于TPU v5e和TPU v6e，这个数字通常约为8。这意味着，例如，如果 $F$ 是16,384而 $B$ 是32，我们理论上可以进行高达`16384 / (32 * 8) = 64`路模型并行而不会对吞吐量产生显著影响。这假设我们可以完全将KV缓存分片到64路，这很困难：我们将在下面讨论这一点。

对于注意力层，我们也以Megatron风格对 $$W_Q$$ 和 $$W_O$$ 进行模型分片。KV权重相当小，复制它们通常比超过 $K$ 路分片更便宜。

<p markdown=1 class="takeaway">**要点：** 在生成期间，我们唯一的选择是模型并行的变体。我们的目标是移动激活值，而不是更大的KV缓存或参数。当我们的批处理大小较大时，我们进行模型并行直到FLOPs-ICI约束（$F / \alpha$）。当我们的批处理大小较小时，我们可以通过更多模型分片来改善延迟（以适度的吞吐量为代价）。当我们想要进行超过KV头数的模型分片时，我们也可以沿批处理维度对KV进行分片。</p>

### 分片KV缓存

**我们还有一个额外的数据结构需要分片——KV缓存。** 同样，我们几乎总是倾向于避免复制缓存，因为它是注意力延迟的主要来源。为此，我们首先沿头维度（head dimension）对KV进行Megatron分片。这限制在 $K$ 路分片，因此对于头数较少的模型，我们尽可能多地分片头维度，然后沿批处理维度进行分片，即 $\text{KV}[2, B_Z, S, K_Y, H]$。这意味着KV缓存是完全分布式的。

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>图示：</b> 注意力机制的比较：(a) 使用纯模型分片的多头注意力，(b) 使用KV缓存批处理分片的多查询注意力。请注意，我们需要两个额外的AllToAll操作将激活值从模型分片转换为批处理分片，以便它们可以作用于KV缓存。" %}

这样做的代价是每个注意力层需要两次AllToAll操作——一次将Q激活值转换为批处理分片，以便我们可以用批处理分片计算注意力，另一次将批处理分片的注意力输出转换回纯模型分片。

{% details 这里是完整的算法！ %}

这里我们将写出完整的注意力算法，其中在 $Y$ 和 $Z$ 两个维度上进行模型并行。我为同时使用 $K$ 表示关键张量和KV头维度表示歉意。设 $M=N/K$。

<div markdown=1 class="algorithm">

1. X[B, D] = ... (现有激活值，来自前一层，未分片)
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K<sub>Y</sub>, H] = ... (现有KV缓存，批处理分片)
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>, M])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

这相当复杂，但你可以大致看出它是如何工作的。新的通信操作开销适中，因为它们作用于我们较小的激活值，而作为回报，我们节省了加载KV缓存（它们是静止的）的大量内存带宽。

</div>

{% enddetails %}

* **序列分片：** 如果批处理大小太小，或者上下文很长，我们可以对KV缓存进行序列分片（sequence sharding）。同样，我们需要支付跨分片累积注意力的集体通信成本。首先，我们需要对Q激活值进行AllGather，然后以类似于Flash Attention的方式累积KV。
## Designing an Effective Inference Engine
迄今为止，我们已经分别探讨了如何高效优化和分片单个预填充与生成操作。要真正有效使用它们，我们需要设计一个推理引擎，能够在我们选择的延迟/吞吐量帕累托前沿上的某一点，同时为这两种操作提供支持。

最简单的方法是运行一批预填充，然后运行一批生成：

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>Figure:</b> 在最简单的设置中，请求被聚合，服务器交替运行一批预填充并调用生成函数，直到所有序列完成。" %}

这易于实现，也是大多数代码库中的第一个推理设置，但它有多个缺点：

1.  **延迟很糟糕。** 我们耦合了预填充和生成批处理大小。首个令牌时间（TTFT）在大批预填充大小时表现很差——你需要完成所有预填充，用户才能看到任何令牌。生成吞吐量在小批量大小时也很差。
2.  **较短的生成会被较长的生成阻塞。** 许多序列会先于其他序列完成，在生成期间留下空的批处理槽位，进一步损害生成吞吐量。随着批处理大小和生成长度的增加，问题会加剧。
3.  **预填充被填充。** 预填充会被填充到最长序列长度，浪费大量计算资源。对此有一些解决方案，但从历史上看，XLA使得跳过这些浮点运算（FLOPs）相当困难。同样，批处理大小和预填充序列长度越大，情况越糟。
4.  **我们被迫在预填充和生成之间共享分片策略。** 预填充和生成都位于同一个切片上，这意味着我们为两者使用相同的拓扑和分片策略（除非保留两份权重副本），这通常对性能无益，例如，生成通常需要更多的模型分片。

因此，该方法仅推荐用于边缘应用（通常只关心服务单个用户并使用浮点运算/字节较少的硬件），以及Transformer代码库生命周期早期的快速迭代（因其简单性）。

一种稍好的方法是在批处理大小为1（此时受计算限制但延迟尚可）的情况下执行预填充，但在生成期间将多个请求批量处理：

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

这将避免批量预填充造成的TTFT浪费，同时保持高生成吞吐量。我们称之为**交错**配置，因为我们"交错"了预填充和生成步骤。这对于像评估这样以吞吐量为主要目标的批量生成应用非常强大。编排器可以配置为在任何生成槽位空出时优先执行预填充，即使对于非常大的生成批处理大小也能确保高利用率。我们也可以避免将预填充填充到最大长度，因为它不是与另一个请求一起批处理的。

主要缺点是，当服务器正在执行预填充时，所有其他请求的生成会暂停，因为所有计算资源都将被预填充消耗。正在解码响应的用户A会被正在预填充的用户B阻塞。这意味着尽管TTFT有所改善，令牌生成平均而言会变得不稳定且缓慢，这对许多应用来说并非好的用户体验——其他用户的预填充处于单个请求总延迟的关键路径上。

为了解决这个问题，我们分离解码和预填充。虽然Transformer推理可以在一台服务器上完成，但从延迟角度来看，通常在两组TPU/GPU上执行两个不同的任务更好。预填充服务器生成键值缓存，通过网络发送到生成服务器，生成服务器将多个缓存一起批处理，并为每个缓存生成令牌。我们称之为**"解耦"**服务。

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

这提供了几个优势：

1.  **规模化下的低延迟**：用户的请求永远不会被其他用户的请求阻塞，除非预填充容量不足。请求应立即进行预填充，然后发送到生成服务器，然后立即插入生成缓冲区。如果我们预期会有许多并发请求，我们可以独立于生成服务器的数量来扩展预填充服务器的数量，这样用户就不会在预填充队列中等待过长时间。
2.  **专业化**：通常，预填充和生成的延迟最优参数分片策略/硬件拓扑差异很大（例如，更多模型并行对生成有用但对预填充无用）。限制两种操作使用相同的分片策略会损害两者的性能，而保留两份权重则占用内存。此外，通过将预填充移至其专用服务器，它不需要保存任何键值缓存，除了它当前正在处理的那个。这意味着我们有更多内存可用于历史缓存（参见下一节）或优化预填充延迟。

一个缺点是键值缓存现在需要通过网络传输。这通常是可接受的，但再次为减少键值缓存大小提供了动机。

<p markdown=1 class="takeaway">**要点总结：** 对于延迟敏感、高吞吐量的服务，我们通常必须将预填充和生成分离到不同的服务器中，预填充以批处理大小1运行，生成则将许多并发请求一起批处理。</p>

### 连续批处理

上述问题（2）引出了**连续批处理**的概念。我们优化并编译：

* 一个预填充函数，处理可变的上下文长度，并将结果插入到具有最大批处理大小和上下文长度/页数的键值缓冲区中。
* 一个生成函数，接收键值缓存，并为所有当前活跃的请求执行生成步骤。

然后，我们将这些函数与一个编排器结合使用，该编排器将传入请求排队，根据可用的生成槽位调用预填充和生成，处理历史缓存（参见下一节），并将令牌流式传输出去。

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### 前缀缓存

由于预填充计算量大且受计算限制（留给我们更少的优化空间），降低其成本的最佳方法之一是减少执行量。因为大语言模型是自回归的，查询["I", "like", "dogs"]和["I", "like", "cats"]产生的键值缓存在前两个令牌上是相同的。这意味着，原则上，如果我们先计算"I like dogs"的缓存，然后计算"I like cats"的缓存，我们只需要执行1/3的计算。我们可以通过重用缓存来节省大部分工作。这在几个特定情况下尤其强大：

1.  **聊天机器人**：大多数聊天机器人对话涉及严格追加的往复对话。这意味着如果我们能保存每轮对话的键值缓存，我们可以跳过除最新令牌以外的所有计算。
2.  **少样本提示**：如果我们有任何形式的少样本提示，这可以被保存并无偿重用。系统指令通常也具有这种形式。

唯一难以实现的原因是内存限制。正如我们所见，键值缓存很大（通常很多GB），为了使缓存有用，我们需要在后续查询到达之前保留它们。通常，预填充服务器上任何未使用的高带宽内存（HBM）都可以用于本地缓存系统。此外，加速器通常在其CPU主机上拥有大量内存（例如，一个8xTPUv5e服务器拥有128GiB的HBM，但大约有450GiB的主机DRAM）。这种内存比HBM慢得多——通常太慢而无法执行生成步骤——但对于缓存读取来说足够快。在实践中：

* 由于键值缓存对于处理初始请求的那组TPU是本地的，我们需要某种形式的亲和性路由，以确保后续查询到达相同的副本。这可能会导致负载平衡问题。
* 更小的键值缓存同样有帮助——它使我们能够在相同的空间内保存更多的键值缓存，并减少读取时间。
* 键值缓存及其查找可以很自然地存储在树或前缀树中。淘汰可以基于最近最少使用（LRU）原则进行。

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>Figure:</b> 实现为LRU前缀树的KV前缀缓存。我们可以通过共享前缀来避免重复的KV内存。来源：<a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai博客</a>。" %}

### 让我们看一个实现：JetStream

Google已经开源了一个实现此逻辑的库，名为[JetStream](https://github.com/google/JetStream)。服务器有一组"预填充引擎"和"生成引擎"，通常位于不同的TPU切片上，由一个单独的控制器协调。预填充发生在"[预填充线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)"中，而生成发生在"[生成线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)"中。我们还有一个"[传输线程](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)"，负责协调将键值缓存从预填充切片复制到生成切片。

Engine接口（在此处[实现](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)）是任何大语言模型必须提供的通用接口。关键方法包括：

*   **prefill：** 接收一组输入令牌并生成一个键值缓存。
*   **insert：** 接收一个键值缓存并将其插入正在生成的键值缓存批次中。
*   **generate：** 接收一组批处理的键值缓存，并为每个批次条目生成一个令牌，将单个令牌的键值缓存追加到每个令牌的解码状态中。

我们还提供了一个JetStream的PyTorch版本，可在[此处](https://github.com/google/jetstream-pytorch)获取。
## Worked Problems
我将基于LLaMA-2 13B为此部分构建一个新模型。详情如下：

| 超参数             | 值     |
| :----------------- | :----- |
| L (层数)           | 64     |
| D (模型维度)       | 4,096  |
| F (前馈网络维度)   | 16,384 |
| N (注意力头数)     | 32     |
| K (键值头数)       | 8      |
| H (查询键值维度)   | 256    |
| V (词表大小)       | 32,128 |

**问题1：** 上述模型有多少参数？在int8精度下，每个token的KV缓存大小是多少？*可以假设我们共享输入和输出投影矩阵。*

{% details 点击此处查看答案。 %}

**参数计数：**

* MLP 参数计数：$L * D * F * 3$
* 注意力机制参数计数：$L * 2 * D * H * (N + K)$
* 词表参数：$D * V$ （因为我们共享这些矩阵）

因此，我们的总参数量为 $L * D * (3F + 2H * (N + K)) + D * V$。代入上述数值，我们得到 `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`。因此，该模型大约有184亿个参数。

KV缓存大小为 int8 精度下每个 token $2 * L * K * H$，即每个 token `2 * 64 * 8 * 256 = 262kB`。

{% enddetails %}

**问题2：** 假设我们要在 TPUv5e 4x4 切片上部署此模型，并且可以在此拓扑上完全分片我们的 KV 缓存。如果我们对所有部分都使用 int8 精度，并希望支持 128k 长度的序列，最大批大小是多少？如果我们把 KV 头数减少到1呢？

{% details 点击此处查看答案。 %}

我们的 KV 缓存大小为 int8 精度下每个 token $2 \cdot L \cdot K \cdot H$，即 `2 * 64 * 8 * 256 = 262kB`。对于 128k 长度的序列，这意味着每个批次条目需要 `262e3 * 128e3 = 33.5GB`。由于每个 TPU 有 16GB 的 HBM（包括我们的参数），我们能容纳的最大批大小是 `(16 * 16e9 - 18.4e9) / 33.5e9 = 7`。如果 $K=1$，那么我们将有8倍于此的容量，即大约56。

{% enddetails %}

**问题3：** 假设参数完全分片在 TPU v5e 4x4 切片上，将它们全部从 HBM 加载到 MXU 需要多长时间？假设参数为 int8 精度。*这是每步延迟的一个很好的下界估计。*

{% details 点击此处查看答案。 %}

我们总共有 184 亿个参数，在 int8 精度下为 18.4e9 字节。每个芯片的 HBM 带宽为 8.2e11，因此如果我们能充分利用 HBM 带宽，大约需要 `18e9 / (8.2e11 * 16) = 1.4ms`。

{% enddetails %}

**问题4：** 假设我们想使用 int8 FLOPs 和参数/激活值，在 TPUv5e 4x4 切片上部署此模型。我们应如何为预填充和解码阶段进行分片？*提示：也许可以先回答这些问题：*

1. 在 4x4 上，ICI 看起来是怎样的？
2. 张量并行的 roofline 约束是什么？
3. 我们如何分片 KV 缓存？

对于此分片方案，生成的每步延迟大约是多少？

**问题5：** 让我们假设上述模型实际上是一个 MoE。MoE 模型本质上是一个拥有 E 份前馈网络块副本的密集模型。每个 token 经过其中 k 个前馈网络块，然后对这 `k` 个块的输出取平均以产生最终输出。我们使用 `E=16` 和 `k=2`，其他设置同上。

1. 它有多少总参数和激活参数？*激活参数指任何给定 token 所使用的参数。*
2. 在 TPU v5e 上，需要多大的批大小才能达到 FLOPs 瓶颈？
3. 每个 token 的 KV 缓存有多大？
4. 包含 T 个 token 的前向传播涉及多少 FLOPs？

{% details 点击此处查看答案。 %}

(1) 作为 MoE，每个 MLP 块现在有 $3 * E * D * F$ 个参数，比密集模型增加了 $E$ 倍。因此，它现在有 $L * D * (3EF + 2H * (N + K)) + D * V$ 或 `64 * 4096 * (3*16*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 212e9` 个总参数，增加了约12倍。对于激活参数，我们有 $k$ 而非 $E$ 个激活参数，总计 `64 * 4096 * (3*2*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 31.2e9`，比密集模型增加了不到2倍。

(2) 因为我们有 $E$ 倍的参数，却只多了 $k$ 倍的 FLOPs，我们的 HBM roofline 增加了 $E/k$ 倍。这意味着在 TPU v5e 上，我们需要大约 `240 * (16 / 2) = 1920` 个 token。

(3) KV 缓存大小保持不变，因为 MoE 的特性并未改变注意力机制的任何方面。

(4) 这仍然是 $2 \cdot \text{激活参数量} \cdot T$。因此是 $2 * \text{31.2e9} * T$。

{% enddetails %}

**问题6：** 对于 MoE，我们可以进行“专家分片”，即在我们的网格的某个轴上拆分专家。用我们的标准记法，我们的第一个 FFW 权重形状为 `[E, D, F]`，我们将其分片为 [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]，其中 `X` 仅在训练期间作为我们的 FSDP 维度使用。假设我们要在 TPU v5e 上进行推理：

1.  在 Y=8, Z=16 的 TPU v5e 8x16 切片上，加载上述模型的 HBM 权重需要多长时间？每个 TPU 有多少可用的 HBM？
2.  我们能部署此模型的最小切片是什么？

**问题7 [二维模型分片]:** 在这里，我们将推导 [ESTI 论文](https://arxiv.org/pdf/2211.05102) 中称为二维权重驻留分片的数学原理。我们在附录 B 中对此有简要描述，但请先尝试解决这个问题，看看你是否能推导出数学过程。二维权重驻留分片的基本思想是将我们的权重沿着 $D$ 和 $F$ 两个轴进行分片，使得每个分块大致是方形的。这减少了通信负载，并允许我们扩展得更远。

以下是二维权重驻留的算法：

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U<sub>X</sub>})
4.  Out[B, D<sub>X</sub>] {U<sub>YZ</sub>} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U<sub>YZ</sub>})
</div>

你的目标是推导出该算法的 $T_\text{math}$ 和 $T_\text{comms}$，并找出它在何时会优于传统的三维模型分片？

{% details 点击此处查看答案！ %}

我们来推导 $T_\text{math}$ 和 $T_\text{comms}$。我们的所有 FLOPs 都被完全分片了，所以和之前一样，我们有 $T_\text{math} = 4BDF / (N \cdot C)$，但我们的通信现在变为

$$\begin{align*}
T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

这里我们注意到 AllReduce 的开销是之前的两倍，并且我们根据每个操作执行的轴数来缩放通信量。假设我们可以自由选择拓扑结构，并且假设 $F=4D$（如 LLaMA-2 中那样），我们声称（通过一些基本的微积分）$X$, $Y$, 和 $Z$ 的最优值是 $X = \sqrt{N / 8}$, $YZ = \sqrt{8N}$，因此总通信量为

$$T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

首先，引用上文，普通的1D模型并行将具有 $T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$，那么新的通信量何时更小呢？我们有

$$\begin{align*}
T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 72
\end{align*}$$

对于一般的 $F$，我们声称这个条件是

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

这告诉我们，如果我们有超过72个芯片，使用这个新方案会更好。这是一个稍微有点奇怪的结果，因为我们过去发现大约在 ~20 路张量并行时就会遇到 ICI 瓶颈。但在这里，即使我们是通信受限的，我们的总通信量仍然随着总芯片数的增加而继续减少！这告诉我们，我们可以继续增加芯片数量，增加批大小，进行更多的参数扩展，并看到延迟的降低。

{% enddetails %}

<h3 markdown=1 class="next-section">第7部分到此结束！关于第8部分，探讨我们如何在TPU上部署LLaMA 3，请点击[此处](../applied-inference)。</h3>
## Appendix
### 附录 A：批处理大小 > 240 的规则有多真实？

我们上面提供的简单规则——批处理大小（batch size）必须大于 240 个标记（tokens）才能受计算限制——大致是正确的，但忽略了 TPU 在某些操作（例如设备间通信）未充分利用全部高带宽内存（HBM）时，能够预取权重的能力。

以下是一个经验图表，显示了一个小型 Transformer 模型每层的时间（微秒）。该模型的 d<sub>model</sub> 为 8192，d<sub>ff</sub> 为 32768，每层只有 2 个矩阵乘法（matmuls）。数据来源于[这个 Colab 笔记本](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)。可以看到，步长时间（step time）在批处理大小约为 240 之前增加非常缓慢，之后则线性增加。

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

以下是实际的吞吐量，单位为 标记/微秒（tokens / us）。这相当清晰地说明了问题。由于我们的层大约有 6 亿参数，这里采用了 4 路分片（sharded），我们预期最小延迟约为 365 微秒。

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

因此，至少在这个模型中，我们确实观察到吞吐量增加，直到每个数据并行分片（data parallel shard）的批处理大小约为 BS240。

### 附录 B：二维权重驻留分片

随着拓扑结构的增长，如果我们能够访问更高维度的网格（如 TPU 的网格），则可以通过引入第二个分片轴来进一步优化，这被称为“**二维权重分片**”。我们称之为“**二维权重驻留**”，在[《高效扩展 Transformer 推理》论文](https://arxiv.org/abs/2211.05102)中有更详细的描述。

因为在 Megatron 中我们只对隐藏层维度 $$F$$ 进行分片，所以当芯片数量因一维分片而大幅增长时，$$F$$ 可能变得显著小于 $$E$$（即 $$d_\text{model}$$ 维度）。这意味着在较大的批处理大小下，在应用 MLP 第一层后，对隐藏维度执行部分集合通信（collectives）可能更为经济。

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

此图显示：

1. 一维权重驻留分片，也称为纯 Megatron 分片，其中激活值（activations）在 AllGather 后被完全复制，权重在隐藏层 F 维度上被完全分片。
2. 二维权重驻留分片，其中权重在隐藏层 F 维度和归约 E 维度上都被分片，激活值在 E 维度上被分片。我们在第一层之前对 (yz) 轴执行 AllGather，然后对 (x) 轴执行 ReduceScatter。

对于注意力层（attention layer），Megatron 风格的分片在芯片数量较少时也相对简单。然而，Megatron 操作发生在 $$n_\text{heads}$$ 维度上，这限制了可能的分片量。通过修改注意力层的二维分片（不是分片隐藏维度，而是分片 $$n_\text{heads}$$ 维度），我们获得了进一步扩展的能力。

### 附录 C：延迟受限的通信

回顾一下，在[第 3 节](../sharding)中，我们推导了在 X 个芯片上通过 1D 环形拓扑（链路为全双工带宽 WICI，延迟 Tmin）执行 AllGather 操作（将数据放入每个 TPU 上大小为 B 的张量中）所需的时间。

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

对于较大的 B，挂钟时间（wall clock）保持相对恒定，因为随着系统中添加更多芯片，执行操作所需的数据移动量和可用的总带宽同时增加。

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

由于在延迟优化推理（latency optimized inference）期间移动的数据量相对较少，激活值上的集合通信通常受延迟项限制（尤其是在小批量大小下）。可以通过计算完成操作所需的跳数（hops）来非常直观地理解延迟。

在 TPU 上，如果通信中与张量大小相关的部分每跳（hop 是指两个相邻设备之间的通信）小于 1 微秒，则可能会受到实际调度集合操作的固定开销的瓶颈限制。在 `4.5e10` 的单向 ICI 带宽下，当满足以下条件时，ICI 通信会变得延迟受限：$$(\text{bytes} / n_\text{shards}) / 4.5e10 < 1e-6$$。对于 8 路 Megatron 分片，当 `buffer_size < 360kB` 时发生。**在推理期间，这实际上并不算小：** 对于 `BS=16` 且 `D=8192`（int8 精度），我们的激活值将使用 `16*8192=131kB`，因此我们已经处于延迟受限状态。

<p markdown=1 class="takeaway">**要点：** 当 $$\text{总字节数} < W_{ICI} \times 1e-6$$ 时，我们的通信变得延迟受限。例如，在 $$Y$$ 维度上进行模型并行（model parallelism）时，当使用 int8 精度且 $$Y > BD / 45,000$$ 时，我们将变得受限。</p>

这里可以与计算屋顶线（compute roofline）画一个平行对比——我们正在承担一些小型操作的固定成本（通信的延迟，矩阵乘法的内存带宽）。

### 附录 D：推测采样

当我们*真正*关心端到端延迟时，可以采用一种额外的技巧，称为推测采样（speculative sampling）<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>。回顾一下，我们通常是从一个大型 Transformer 中逐个生成标记：

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

使用推测采样时，我们使用一个更小、更廉价的模型来生成标记，然后用大模型检查结果。这最容易通过*贪心解码（greedy decoding）* 来理解：

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. 我们从某个更小、更廉价的模型中贪心地采样。理想情况下，我们使用经过训练以匹配大模型的模型（例如通过蒸馏），但也可以简单地使用 n-gram 或在一个小型文本语料库上进行标记匹配。
2. 在我们生成 K 个标记后，我们使用大模型计算迄今为止生成的所有标记的下一个标记对数概率（logits）。
3. 由于我们是贪心解码，我们可以直接检查较小模型生成的标记是否在所有可能标记中具有最高概率。如果某个标记是错误的，我们取最长的正确前缀，用正确标记替换第一个错误标记，然后返回步骤 (1)。如果所有标记都正确，我们可以使用最后一个正确的对数概率来采样一个额外标记，然后再返回步骤 (1)。

**为什么这是延迟上的胜利？** 这个方案仍然需要我们为每个标记执行相当于一次大模型前向传播的计算量，但因为我们能将一批标记一起处理，所以可以在一次前向传播中完成所有这些计算，并利用我们*不* *受计算限制*这一事实来“免费”评估更多标记。

平均而言，每个被接受的标记在 FLOPs 方面成本更高（因为有些会被拒绝，而且我们必须调用草稿模型），但我们从硬件中榨取了更多的 FLOPs，并且小模型成本低廉，所以总体上我们是赢家。我们还在多个步骤中共享 KV 缓存加载，因此**推测解码对于长上下文也可以是吞吐量上的胜利。** 由于所有内容都经过了大模型的检查，我们完全不会改变采样分布（尽管对于非贪心情况，精确的轨迹会有所不同）。

传统上，推测解码依赖于存在一个与目标模型具有相似采样分布的小型模型，例如用 LLaMA-2 2B 作为 LLaMA-2 70B 的草稿模型，但这通常不存在。即使存在，如果接受率较低，较小的草稿模型可能仍然太昂贵。相反，将草稿模型嵌入主模型中会有所帮助，例如通过在基座模型的后期层中添加一个专门的草稿头（drafter head）<d-cite key="eagle"></d-cite><d-cite key="medusa"></d-cite><d-cite key="DeepSeek3"></d-cite>。由于这个头与主模型共享大部分参数，因此运行速度更快，并且能更紧密地匹配采样分布。

对于普通的自回归采样，每秒标记数（token/s）与步长时间（step time）相同。我们仍然受限于根据算术强度（Arithmetic Intensity）部分所述的理论最小步长时间（事实上，推测采样的步长时间通常比普通自回归采样慢得多，但因为我们平均每个步骤能获得超过 1 个标记，所以我们可以获得更好的 tokens/s）。

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>图：</b> 此图显示了 Chinchilla（DeepMind 的 70B 模型）搭配一个 4B 参数草稿模型（小模型）时的每步延迟和推测成功率。对于 XSum（一个自然语言数据集），理想的推测量大约是提前 3-4 个标记，而 HumanEval（一个代码数据集）更可预测，从更激进的推测中获益。" %}

**这对非贪心解码如何工作？** 这稍微复杂一些，但本质上归结为一种受 Metropolis-Hastings 启发的算法，其中我们有从对数概率得出的 $$P_{\text{draft model}}(\text{chosen token})$$ 和 $$P_{\text{target model}}(\text{chosen token})$$，如果这些概率的比率小于某个阈值，则会以一定概率拒绝所选标记。

这两篇[论文](https://arxiv.org/abs/2211.17192)[同时](https://arxiv.org/abs/2302.01318)推导出了这一点，并有很好的例子说明其在实践中的工作原理。

<p markdown=1 class="takeaway">**要点：** 推测采样是用吞吐量换取更好每标记延迟的另一个强大杠杆。然而，在批处理大小受限的场景下（例如硬件占用小或 KV 缓存大），它成为了一个双赢的选择。</p>