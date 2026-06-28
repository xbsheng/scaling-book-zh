---
layout: distill
title: "All the Transformer Math You Need to Know"
# permalink: /main/
description: "Here we'll do a quick review of the Transformer architecture, specifically how to calculate FLOPs, bytes, and other quantities of interest."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: "../sharding"
previous_section_name: "Part 3: Sharding"

next_section_url: ../training
next_section_name: "Part 5: Training"

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

bibliography: main.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "Counting Dots"
  - subsections:
    - name: "Forward and reverse FLOPs"
  - name: "Transformer Accounting"
  - name: "Global FLOPs and Params Calculation"
  - name: "Miscellaneous Math"
  - subsections:
    - name: "Sparsity and Mixture-of-Experts"
    - name: "Gradient checkpointing"
    - name: "Key-Value (KV) caching"
  - name: "What Should You Take Away from this Section?"
  - name: "A Few Problems to Work"
  - name: "Appendix"
  - subsections:
    - name: "Appendix A: How does Flash Attention work?"

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

## Counting Dots
首先考虑以下向量 $$x$$、$$y$$ 和矩阵 $$A$$、$$B$$，其形状如下：

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{数组} & \textrm{形状} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end{array}
$$

- 向量点积 $$x \cdot y$$ 需要 $$P$$ 次_加法_和_乘法_运算，即总共 $$2P$$ 次浮点运算。
- 矩阵-向量乘积 $$Ax$$ 沿矩阵 $$A$$ 的每一行进行 $$N$$ 次点积运算，共需 $$2NP$$ 次浮点运算。
- 矩阵-矩阵乘积 $$AB$$ 针对矩阵 $$B$$ 的 $$M$$ 列中的每一列执行一次矩阵-向量乘积，共需 $$2NPM$$ 次浮点运算。
- 一般而言，如果我们有两个更高维度的数组 $$C$$ 和 $$D$$，其中某些维度是<span style="color:red">收缩维度</span>，某些维度是<span style="color:blue">批处理维度</span>（例如 $$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$），则此收缩操作的浮点运算量是 $$C$$ 和 $$D$$ 所有维度乘积的两倍，其中批处理维度和收缩维度仅计算一次（例如 $$2\blue{GH}IJMN\red{KL}$$）。注意，一个维度只有同时出现在两个乘法操作数中时才是批处理维度。（还需注意，如果没有收缩维度且仅为逐元素乘积，则系数 2 不适用。）<d-footnote><b>收缩维度</b>是操作中进行求和的轴（它们在输入中均出现，但在输出中不出现），例如矩阵乘法中的内维。<b>批处理维度</b>是共享的轴，在输入中均出现并原样传递到输出；它们索引独立的子问题，在浮点运算计数中不相乘。用爱因斯坦求和约定的术语来说：在输入和输出中都出现的标签是批处理维度；在输入中出现但在输出中缺失的标签是收缩维度。</d-footnote>

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{浮点运算量} & \textrm{数据量} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{批处理} \\ d_j \notin \red{收缩}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end{array}
$$

请注意一个事实：对于矩阵-矩阵乘法，*计算量*以立方级 $$O(N^3)$$ 缩放，而数据传输量仅以平方级 $$O(N^2)$$ 缩放——这意味着随着矩阵乘法规模的扩大，*更容易*达到计算饱和的极限。这是极不寻常的，并在很大程度上解释了为何我们使用以矩阵乘法为主导的架构——它们具有良好的可扩展性！

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### 前向和反向浮点运算量

在训练过程中，我们并不特别关注给定矩阵乘法的结果；我们真正关心的是其导数。这意味着在反向传播过程中，我们会执行显著更多的浮点运算。

假设 **B** 只是更大网络中的一个矩阵，**A** 是我们的输入激活值，且 **C = A B**，则损失 **L** 对 **B** 的导数由链式法则给出：

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

计算它需要 $$2NPM$$ 次浮点运算（因为它在 $$N$$ 维度上进行收缩）。同样，损失对 **A** 的导数为

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

这同样是 $$2NPM$$ 次浮点运算，因为 **dL/dC** 是大小为 $$[N, M]$$ 的矩阵。虽然这个量不是关于参数的导数，但它用于计算网络前序层的导数（例如，就像上面的 dL/dC 用于计算 dL/dB 一样）。

将这些加总起来，我们发现**在训练过程中，总共需要 6NPM 次浮点运算**，而推理过程中仅需 2NPM 次：前向传播 2NPM 次，反向传播 4NPM 次。由于 PM 是矩阵中的参数数量，这就是著名的 $$6 * \text{参数数量} * \text{标记数量}$$ Transformer 训练浮点运算近似公式的最简单形式：每个标记需要 $$6 * \text{参数数量}$$ 次浮点运算。我们将在下文展示一个更准确的推导。
## Transformer Accounting
Transformer是未来的趋势。嗯，至少可以说是当下的主流。也许几年前，它还只是众多架构之一。但如今，了解这个架构的每个细节都至关重要。我们不再重新介绍该架构，但[这篇博客](https://jalammar.github.io/illustrated-transformer/)和[原始Transformer论文](https://arxiv.org/abs/1706.03762)可能成为有用的参考。

这是Transformer解码器架构的基本示意图：

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图示：</b>此图展示标准Transformer的一层结构，信息从上至下流动。我们采用单字母约定来描述Transformer中数组的形状与布局，收缩维度用红色表示，批次维度用蓝色表示。在特定操作中，输入形状显示于左上方，参数形状显示于右上方，结果形状位于下方，例如BTD是门控爱因斯坦求和(gating einsum)的输入形状，DF是权重形状。" %}

**注 [门控爱因斯坦求和]**：上图使用了一种"门控爱因斯坦求和(gating einsum)"<d-cite key="glu"></d-cite>，我们将升维投影矩阵拆分为两个矩阵（即上图中的$W_\text{In1}$和$W_\text{In2}$），它们的输出通过逐元素相乘形成一种"门控函数"。并非所有大语言模型都采用此设计，因此有时会看到单一的$W_\text{In}$矩阵，此时MLP总参数量为2DF而非3DF。通常情况下，这种设计会通过扩大D和F的维度来保持与三矩阵方案相同的参数量。值得一提的是，LLaMA、DeepSeek及其他许多模型都采用了某种形式的门控爱因斯坦求和。

**注2 [多头注意力机制]**：在自注意力中，T和S相同；而在交叉注意力中，它们可能不同。标准的多头注意力机制(MHA)中，N和K相同；而在[多查询注意力机制](https://arxiv.org/abs/1911.02150)(MQA)<d-cite key="mqa"></d-cite>中K=1；对于[分组多查询注意力机制](https://arxiv.org/abs/2305.13245)(GMQA)<d-cite key="gmqa"></d-cite>，K只需满足能够整除N即可。

**注3 [前归一化与后归一化]**：上图展示的是所谓的"后归一化(post-norm)"Transformer，其中层归一化操作位于残差连接之后，即`norm(x + attn(x))`。这与原始Transformer论文一致，但当今大多数现代Transformer采用"前归一化(pre-norm)"架构，归一化在残差连接前进行，通常表示为`x + attn(norm(x))`。诸如LLaMA-3等模型现采用此设计。
## Global FLOPs and Params Calculation
以下内容我们将计算每层的浮点运算数（FLOPs），以避免在各处都引入 **L** 因子。

### 多层感知器（MLPs）

Transformer 中的多层感知器（MLPs）通常由两次输入矩阵乘法（matmul）经过逐元素组合，以及一次输出矩阵乘法组成：

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{训练浮点运算数} & \textrm{参数量} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

### 注意力机制（Attention）

对于通用的**分组查询注意力（grouped-query attention）**情况，其 **Q** 和 **KV** 的头数量不同。我们假设 **Q**、**K**、**V** 投影的头维度 H 相同，并估算 **QKVO** 矩阵乘法的计算成本：

$$
\begin{array}{ccc}
\textrm{操作} & \textrm{训练浮点运算数} & \textrm{参数量} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

**点积注意力操作（dot-product attention）**更为复杂，它实际上是一个在 $$B$$、$$K$$ 维度上批量进行的 $$TH \cdot HS$$ 矩阵乘法，一次 softmax 操作，以及一个再次在 $$B$$、$$K$$ 维度上批量进行的 $$TS \cdot SH$$ 矩阵乘法。我们用蓝色标注批量处理的维度：

$$
\begin{array}{cc}
\textrm{操作} & \textrm{训练浮点运算数} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H]
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

**注意 [因果掩码（causal masking）]**：大多数最新的 Transformer 使用因果掩码，而不是完整的双向注意力。在这种情况下，点积操作的有效浮点运算数会减少一半。为了在实践中实现这种减少，我们需要使用专门的注意力核（attention kernel），而不是朴素的爱因斯坦求和约定（einsum）。

### 其他操作

Transformer 中还包含其他几种操作。层归一化（Layernorms）的计算成本相对较低，在初步成本估算中可以忽略。此外，最后还有一个巨大的（尽管不是每层都有的）**反嵌入矩阵乘法（unembedding matrix multiply）**。

$$
\begin{array}{ccc}
\textsf{操作} & \textsf{训练浮点运算数} & \textsf{参数量} \\
\hline \\
\textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

### Transformer 浮点运算数的一般经验法则

如果我们忽略短上下文训练中点积注意力的成本，那么所有层的总浮点运算数为：

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{num tokens} * \textrm{parameter count}
\end{align*}
$$

这引出了一个著名的估算密集型 Transformer 浮点运算数的经验法则，忽略了注意力运算的浮点运算数。（反嵌入是另一个简单的矩阵乘法，其浮点运算数为 $6BTDV$，参数量为 $DV$，并遵循相同的经验法则。）

### 注意力成本随上下文长度的占比

如果我们考虑上述的点积注意力，并假设 $$F=4D$$、$$D=NH$$（这是典型情况）以及 $$N=K$$：

$$\small{\frac{\textrm{attention FLOPs}}{\textrm{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

因此，关键的结论是：**只有在 T > 8D 时，点积注意力浮点运算数在训练中才开始占主导地位**。对于 D ~ 8k 的情况，这大约对应于 64K 个词元。这有些道理，因为这意味着随着 MLP 规模的增加，注意力浮点运算数的相对重要性会降低。对于大模型，注意力机制的二次计算成本实际上并不是长上下文训练的巨大障碍。然而，对于较小的模型，即使是例如 Gemma-27B，D=4608，这意味着注意力机制在序列长度约为 37k 时开始占主导地位。<d-footnote>请注意，一些现代开源模型引入了局部注意力或其他优化来降低注意力成本，并改变了这个性能瓶颈。</d-footnote> Flash Attention 也有助于缓解长上下文的成本，我们在[附录 A](#appendix-a-how-does-flash-attention-work)中会简要讨论。
## Miscellaneous Math
### 稀疏性与混合专家模型

我们不得不简要讨论混合专家模型（Mixture of Experts, MoE）<d-cite key="moe"></d-cite>，它用一组可动态路由的独立MLP替代了标准Transformer中单一的密集MLP块。在**第一近似**下，**MoE本质上是一个每层包含E个MLP块的普通密集模型**，而非仅有一个。每个token激活这些专家中的$k$个，通常$k \ll E$。比例$E / k$被称为稀疏度（sparsity），通常介于8到64之间（例如[DeepSeek v3](https://arxiv.org/pdf/2412.19437)有效参数为$k=8$, $E=256$）。相较于密集版本，这使参数量增加$O(E)$倍，同时每个token激活的参数总量仅扩大$k$倍。

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>图示：</b>包含$n$个专家的MoE层示例。门控专家将每个token路由至其中$k$个，这些MLP的输出被求和。我们的参数量是每个专家参数量的$n$倍，但每个token仅使用$k$个。<a href=\"https://deepgram.com/learn/mixture-of-experts-ml-model-guide\">来源</a>。" %}

与密集模型相比，MoE引入了新的通信开销，主要是两次AllToAll操作（分别位于MoE块前后），用于将token路由至正确专家并将其结果回传至原设备。<d-footnote>技术上讲，这仅发生在数据或序列分片沿专家所在轴分布时。</d-footnote> 但如前一节所述，对于双向环状拓扑，单次AllToAll的开销仅为等效AllGather操作的1/4。

### 梯度检查点

反向传播算法以计算换内存。若不要求反向传播需$$O(n_\text{layers}^2)$$次浮点运算（FLOPs），**则要求$$O(n_\text{layers})$$的内存**来保存前向传播生成的所有中间激活值。尽管这优于二次计算开销，但内存消耗惊人：以$$B * T=4M$$（每批400万token）、L=64、D=8192的模型为例，若完全避免冗余反向计算，需保存约$$2 * 20 * B * T * D * L = 84TB$$的bfloat16格式激活值。其中20源于对上述Transformer图中每个中间节点的大致统计，例如：

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

因此为避免重算，需保存前向传播中的$$g(x)$$和$$\exp(g(x))$$。为节省内存，我们可选择仅保存部分中间激活值，常用策略如下：

* **分块重计算（Block remat）**：仅保存每层输入。此策略最为激进，每层仅保存1个检查点，上述案例中仅需保存4.2TB。这迫使我们在反向传播中重复几乎所有前向计算，使FLOPs从$$6 \cdot \text{num params} \cdot \text{num tokens}$$增至约$$8 \cdot \text{num params} \cdot \text{num tokens}$$。
* **仅保存大矩阵乘法结果**：另一种简单策略是仅保存大型矩阵乘法的输出。这避免了反向传播中大型矩阵乘法的重算，但仍需重算其他激活函数及注意力机制部分。这将每层20个中间值减少至约7个。

此列举并不全面。在JAX框架中，这些策略通常通过`jax.remat`/`jax.checkpoint`控制（详见[此处文档](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)）。

### 键值缓存（KV Cache）

如[第7章](../inference)所述，LLM推理包含两个关键阶段：预填充（prefill）与生成（generation）。

* **预填充**处理长提示（prompt），并在键值缓存（Key-Value Cache, KV Cache）中保存其注意力激活值（具体为注意力模块中的键值投影）以供生成阶段使用。
* **生成**将多个KV缓存批量处理，并从中采样token。

每个KV缓存本质上是尺寸为$[2, S, L, K, H]$的数组，其中维度2对应键（keys）和值（values）。其规模相当庞大！int8精度的KV缓存总量为$2SLKH$。以中等规模模型（8k上下文长度、64层、$KH = NH = D = 8192$）为例，需$2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$空间。这解释了为何需采用$K \ll N$的分组查询注意力（GMQA）机制。
## What Should You Take Away from this Section?
Transformer 的整体参数和 FLOPs（浮点运算数）相对容易计算，在此进行总结，假设使用 MHA（多头注意力机制）（批大小 B、词汇表大小 V、序列长度 T、D=d<sub>model</sub>、F=d<sub>ff</sub>）：

<!-- $$
\begin{array}{ccc}
\textrm{Component} & \textrm{Params per layer} & \textrm{Training FLOPs per layer} \\
\hline \\
\textbf{MLP} & 3DF & 18BTDF \\[10pt]
\textbf{Attention} & 4DNH & 24BTDNH + 12BT^2NH \\[10pt]
\textbf{Other} & D & BTD \\[10pt]
\textbf{Vocab} & DB \text{ (total, not per-layer)} & 12BTDV \\[10pt]
\end{array}
$$ -->

| 组件          | 每层参数量              | 每层训练FLOPs               |
| :------------ | :------------------------ | :---------------------------- |
| **MLP**       | 3DF                       | 18BTDF                        |
| **注意力**    | 4DNH                      | 24BTDNH \+ 12BT<sup>2</sup>NH |
| **其他**      | D                         | BTD                           |
| **词汇表**    | DV (整体，非每层)         | 12BTDV                        |

* MLP（多层感知机）块的参数量主导了总参数量，并且只要序列长度 $T < 8D$，MLP 块也主导了 FLOPs 预算。
* 对于合理的上下文长度，训练期间的总 FLOPs 预算可以通过 $$6 \cdot \text{num_params} \cdot \text{num_tokens}$$ 很好地近似。
* 在推理期间，我们的 KV 缓存（KV caches）每个缓存大致为 $$2 \cdot S \cdot L \cdot K \cdot H$$（其中 K 是 KV 头的数量），尽管架构改进通常可以减小这一开销。
## A Few Problems to Work
**问题 1:** 一个具有 $D=4096$, $F=4 \cdot D$, $V=32,000$, 和 $L=64$ 的模型有多少个参数？其中有多少比例是注意力参数（attention parameters）？每个token的KV缓存（KV caches）有多大？*你可以假设 $N\cdot H=D$ 并且使用int8类型的KV进行多头注意力（multi-head attention）。*

{% details 点击查看答案。 %}

1. 总参数量大约为 $$L \cdot (3DF + 4DNH + D) + 2DV$$。对于给定的数值，这等于 $$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$，即16B（160亿）参数。
2. 注意力参数占总参数量的比例通常是 $$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$。这意味着大约四分之一的参数用于注意力。
3. 每个token，我们的KV缓存大小为 $$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$（以int8计），即 `512kB / token`。

{% enddetails %}

**问题 2:** 在 `{'X': 4, 'Y': 8, 'Z': 4}` 的配置下，执行 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] 总共需要多少FLOPs（浮点运算次数）？每个TPU执行了多少FLOPs？

{% details 点击查看答案。 %}

该操作的总“理论”FLOPs为 $$2 \cdot B \cdot D \cdot F$$。然而，由于计算并未在Z维度上分片，我们实际上执行了Z倍的FLOPs，即总FLOPs为 $$2 \cdot B \cdot D \cdot F \cdot Z$$。由于计算在其他维度上分片，每个设备上的总FLOPs大约为 $$2 \cdot B \cdot D \cdot F / (X \cdot  Y)$$。

{% enddetails %}

**问题 3:** 执行 $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$ 涉及多少FLOPs？

{% details 点击查看答案。 %}

根据上述规则，I和J是缩并维度（contracting dimensions），而K、L、M、N和O是非缩并维度（non-contracting dimensions）。我们没有“批处理维度（batching dimensions）”，因此这只是所有轴尺寸的乘积，即 $$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$。如果存在共享轴，则只计算一次。

{% enddetails %}

**问题 4:** 自注意力（self-attention）的算术强度（arithmetic intensity）是多少？（忽略Q/K/V/O投影）。*请将答案表示为Q和KV长度T和S的函数。* 在什么上下文长度下，注意力是计算受限（FLOPs-bound）的？给定我们TPU的HBM带宽，绘制随着上下文长度增长，注意力相对于FFW块的有效相对成本。

{% details 点击查看答案。 %}

自注意力需要加载 $$Q$$、$$K$$ 和 $$V$$ 激活，然后计算 $$\text{softmax}(Q \cdot K) \cdot V$$，最后将结果写回HBM。这将使用Flash Attention来完成，因此在数学上有一些注意事项，但基本上在bf16格式下，自注意力执行：

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

所以我们的总字节数为 $$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$，总FLOPs为 $$4BTSNH + O(BTSN)$$，算术强度为 $$4BTSKGH / (4BHK * (TG + S))$$。

因此，在预填充（prefill）阶段，我们有 $$S=T$$，所以算术强度为 $$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$。在生成（generation）阶段，$$T=1$$，因此我们有 $$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$（假设 $$S$$ 非常大）。取决于你如何理解这个问题，在预填充或训练期间，假设没有序列分片（sequence sharding），在S=240时自注意力是计算受限的。在生成阶段，由于 $$G$$ 很小，我们永远不是计算受限的。尽管如此，你可以看到增加 $$G$$ 会使我们更接近计算受限。

{% enddetails %}

**问题 5:** 在什么序列长度下，自注意力的FLOPs等于QKVO投影的FLOPs？

{% details 点击查看答案。 %}

这纯粹是一个何时 $$24BTDNH = 12BT^2NH$$ 的问题。简化后我们得到 $$2D = T$$，因此例如对于 $$D=4096$$，这是 $$8192$$。这告诉我们，对于大多数合理的上下文长度，矩阵乘法（matmul）的FLOPs更大。

{% enddetails %}

**问题 6:** 假设在前向传播期间，我们只保存Transformer层中7个主要矩阵乘法的输出（Q、K、V、O 加上三个FFW矩阵）。在反向传播期间，我们需要额外“重新物化（rematerialize）”多少FLOPs？

{% details 点击查看答案。 %}

仅保存七个矩阵乘法输出（Q、K、V、O、W₁、W₂、W₃）意味着反向传播必须重新计算两个注意力矩阵乘法：

$$QK^{\top} \quad\text{和}\quad \operatorname{softmax}(QK^{\top})V$$

以获得 $\frac{\partial L}{\partial W_\text{O}}$。

每一个都是一个在 $B$ 个序列和 $N$ 个头（head）上批处理的 $T \times T$ 矩阵乘法，因此额外的FLOPs为：

$$4 \; B \, T^{2} \, N \, H.$$

其他重新计算的操作是：
1. 对于 $\frac{\partial L}{\partial W_\text{In1}}$ 和 $\frac{\partial L}{\partial W_\text{In2}}$，是 $O(BTD)$。
2. 对于 $\frac{\partial L}{\partial W_\text{Out}}$，是 $O(BTF)$。

{% enddetails %}

**问题 7:** DeepSeek v3 报告称，它使用了2.79M H800小时在14.8T token上进行了训练（[来源](https://arxiv.org/pdf/2412.19437v1)）。鉴于其拥有37B激活参数，他们大约实现了多高的硬件利用率？*提示：注意他们使用了不带结构化稀疏性的FP8 FLOPs。*

{% details 点击查看答案。 %}

根据[此处](https://lenovopress.lenovo.com/lp1814.pdf)的规格表，我们找到带稀疏性的FP8性能为3,026 TFLOPs/s，通常不带稀疏性时为这个数值的一半（`1.513e15` FLOPs/s）。2.79M H800小时意味着 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` 总FLOPs。给定37B的激活参数计数，这次训练运行应该使用了大约 `6 * 37e9 * 14.8e12 = 3.3e24` FLOPs。这意味着FLOPs利用率大约为 `3.3e24 / 1.52e25 = 21.7%`。

{% enddetails %}

**问题 8:** 混合专家模型（Mixture of Experts, MoE）拥有 $E$ 个标准稠密MLP块（dense MLP block）的副本，每个token激活其中的 $k$ 个专家。对于权重为int8的MoE，在TPU v5e上，需要多大的token批次大小（batch size）才能达到计算受限？对于拥有256个（路由）专家（routed experts）且 $k=8$ 的DeepSeek，这个数字是多少？

{% details 点击查看答案。 %}

因为我们有 $E$ 个专家的副本，在int8下，对于每个权重矩阵，我们需要加载 $E \cdot D \cdot F$ 字节。由于每个token激活 $k$ 个专家，对于每个权重矩阵，我们有 $2\cdot k \cdot B \cdot D \cdot F$ FLOPs。为了在int8权重和bfloat16 FLOPs下达到计算受限，我们需要算术强度（每加载字节的FLOPs）超过TPU的约240 FLOPs/字节，这发生在 $(2\cdot k \cdot BDF) / EDF > 240$ 或 $k \cdot B / E > 120$ 时。

因此，我们需要 $B > 120 \cdot E / k$ 来达到计算受限。对于DeepSeek，这给出 $B > 120 \cdot 256 / 8 = 3840$。这是一个在生成时非常大的批次大小。

{% enddetails %}

<h3 markdown=1 class="next-section">第四部分到此结束！第五部分（关于Transformer训练的扩展），[请点击这里](../training)！</h3>
## Appendix
### 附录A：Flash Attention 如何工作？

传统上反对将Transformer扩展到非常长上下文的理由是，注意力机制的浮点运算次数（FLOPs）和内存使用量会随着上下文长度的增加呈二次方增长。虽然注意力机制的QK点积的形状确实是$[B, T, S, N]$，其中B是批次大小（batch size），T和S分别是Q和K序列的维度，N是头的数量（number of heads），但这一说法带有一些重要的注意事项：

1. 正如我们之前指出的，尽管是二次方关系，但注意力机制的FLOPs只有在$$T > 8 \cdot D$$时才占主导地位，特别是在训练期间，与内存中的所有权重和激活检查点（尤其是经过分片时）相比，单个注意力矩阵的内存占用很小。
2. 我们无需实例化完整的注意力矩阵来计算注意力！我们可以计算局部和与最大值，避免实例化超过一小块数组。虽然总FLOPs仍是二次方，但我们大幅降低了内存压力。

第二个观察首先由 [Rabe et al. 2021](https://arxiv.org/abs/2112.05682) 提出，后来在 [Flash Attention论文](https://arxiv.org/abs/2205.14135) (Dao et al. 2022) 中得以发展。其基本思想是按K/V块（chunk）计算注意力，其中我们计算局部softmax和一些辅助统计量，然后将它们传递给下一个块，下一个块再将其与自身的局部块结合起来。具体来说，我们计算：

1. **M：** $$q \cdot k$$ 沿序列维度的运行最大值
2. **O：** 沿序列维度的运行完整注意力softmax
3. **L：** 运行的分母 $$\sum_i \exp(q \cdot k_i - \text{运行最大值})$$

有了这些，我们只需要常数级别的内存就可以计算新的最大值、新的运行和以及新的输出。粗略描述其工作原理，注意力机制大致是以下运算：

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

为了数值稳定性而减去最大值，可以将其加回来而不影响结果，因为 $$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$。仅看上面的分母，如果我们假设有两个连续的键向量块 $$K^1$$ 和 $$K^2$$，并且我们为每个块计算了局部softmax和 $$L^1$$ 和 $$L^2$$：

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^2)$$

那么我们可以使用以下公式将它们合并为这两个块组合起来的完整softmax和：

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

其中

$$M^1 = \max_j Q \cdot K_j^1 \text{ 且 } M^2 = \max_j Q \cdot K_j^2$$

这种方法也可以用于完整的softmax，从而为我们提供了一种累积任意大小softmax和的方法。这里是Flash Attention论文中的完整算法。

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

从硬件角度来看，这使得我们可以将Q的块放入VMEM（上述算法中称为片上SRAM），因此我们只需要在每次迭代时加载KV块，从而提高了算术强度（arithmetic intensity）。我们还可以将运行统计量保存在VMEM中。

最后值得强调的一个微妙之处是注意力softmax的一个属性，该属性被用于使Flash VJP（反向模式导数）的计算对训练来说切实可行。如果我们定义一个中间softmax数组为：

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_l e^{\tau q_i \cdot k_l}}$$

在注意力机制中，我们从反向模式的 *dO* 和 *V* 数组获得 *dS*：

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

在将这个梯度反向传播到Q和K时，

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

我们利用一个恒等式，允许我们将沿大型键**长度**维度的收缩与沿特征**深度**维度的局部收缩进行交换。

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}$$

这个替换对于能够为VJP实现序列块*局部*计算至关重要，并使得进一步的巧妙分片方案（如环形注意力）成为可能。