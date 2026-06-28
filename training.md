---
layout: distill
title: "How to Parallelize a Transformer for Training"
# permalink: /main/
description: "Here we discuss four main parallelism schemes used during LLM training: data parallelism, fully-sharded data parallelism (FSDP), tensor parallelism, and pipeline parallelism. For each, we calculate at what point we become bottlenecked by communication."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "../transformers"
previous_section_name: "Part 4: Transformers"

next_section_url: ../applied-training
next_section_name: "Part 6: Training LLaMA"

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
  - name: "What Do We Mean By Scaling?"
  - subsections:
    - name: "Data Parallelism"
    - name: "Fully-Sharded Data Parallelism (FSDP)"
    - name: "Tensor Parallelism"
    - name: "Combining FSDP and Tensor Parallelism"
    - name: "Pipelining"
    - name: "Scaling Across Pods"
  - name: "Takeaways from LLM Training on TPUs"
  - name: "Some Problems to Work"
  - name: "Appendix"
  - subsections:
    - name: "Appendix A: Deriving the backward pass comms"

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

## What Do We Mean By Scaling?
**模型扩展的目标**是在增加训练或推理所用芯片数量的同时，实现吞吐量的线性比例增长（我们称之为*强扩展性*）。虽然单个芯片的性能取决于内存带宽与浮点运算量之间的权衡，但集群级别的性能则取决于通过将有用计算与芯片间通信重叠执行来隐藏通信开销。这并非易事，因为增加芯片数量会增加通信负载，同时减少可用于隐藏通信的每设备计算量。正如我们在[第3章](../sharding)中看到的，分片矩阵乘法通常需要昂贵的AllGathers或ReduceScatter操作，这可能会阻碍TPU执行有用计算。本节的目标是找出这些操作何时会变得*过于昂贵*。

本节我们将讨论四种常见的并行方案：（纯）**数据并行**、**完全分片数据并行**（FSDP / ZeRO分片）、**张量并行**（也称为模型并行），以及（简要介绍）**流水线并行**。对于每种方案，我们将展示其产生的通信成本，以及该成本在何时开始成为计算成本的瓶颈。<d-footnote>我们将重点关注通信瓶颈——虽然内存容量限制很重要，但在使用重计算（激活检查点）和在预训练阶段使用大量芯片时，它们通常不会成为限制因素。我们在此也不讨论MoE的专家并行——这将极大地扩展设计空间，仅讨论稠密Transformer的基础情况。</d-footnote> 在本节中，您可以只关注芯片间的通信成本，因为只要单个芯片的批处理足够大，数据从HBM到MXU的传输就已经与计算重叠。

为简化整个本节的计算，我们将使用以下符号。

| 符号 | 含义（模型参数）                                             |
| :--- | :----------------------------------------------------------- |
| D    | **d**<sub>model</sub>（隐藏维度/残差流维度）                 |
| F    | **d**<sub>ff</sub>（前馈维度）                               |
| B    | 批处理维度（批次中的标记数；总数，而非每设备数量）           |
| T    | 序列长度                                                     |
| L    | 模型层数                                                     |

| 符号 | 含义（硬件特性）                                                                 |
| :--- | :------------------------------------------------------------------------------- |
| C    | 每芯片FLOPS/s                                                                   |
| W    | 网络带宽（双向，常加下标如 $W_{\text{ici}}$ 或 $W_{\text{dcn}}$）               |
| X    | 沿网格轴X的芯片数量                                                             |
| Y    | 沿另一网格轴（标记为Y）的芯片数量                                               |
| Z    | 沿第三网格轴（标记为Z）的芯片数量                                               |

为简单起见，**我们将近似地将Transformer视为一个MLP块堆栈**——正如我们在[第4章](../transformers)中所看到的，对于较大的模型，注意力机制在浮点运算量中所占比例相对较小。我们还将忽略门控矩阵乘法，从而得到每层的以下简单结构：

{% include figure.liquid path="assets/img/transformer-layer.png" class="img-fluid" caption="<b>图示：</b>一个简化的Transformer层。我们将每个FFW块视为两个矩阵的堆栈：<b>W<sub>in</sub></b>: <code>bf16[D, F]</code>（上投影）和<b>W<sub>out</sub></b>: <code>bf16[F, D]</code>（下投影），输入为<b>In</b>: <code>bf16[B, D]</code>。" %}

{% details 以下是这个简单Transformer在无并行情况下的完整算法。 %}

<div markdown=1 class="algorithm">

**前向传播：** 需要计算 Loss[B]

1.  Tmp[B, F] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B, D] = Tmp[B, F] *<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B, D] = ...
2.  dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
3.  dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
4.  dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
5.  dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*用于前一*)

</div>

此处提供此算法是为了与后续添加了通信的算法进行对比。

{% enddetails %}

以下是我们将讨论的4种并行方案。每种方案可以被认为由上图中**In**、**W<sub>in</sub>、W<sub>out</sub>和Out**的分片方式唯一定义。

**1. 数据并行：** *激活值沿批处理维度分片，参数和优化器状态在每个设备上复制。通信仅发生在反向传播期间。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. 完全分片数据并行（FSDP或ZeRO-3）：** *激活值沿批处理维度分片（类似纯数据并行），参数沿同一网格轴分片，并在前向传播使用前即时进行AllGather。优化器状态也沿批处理维度分片。减少重复内存占用。*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. 张量并行（也称为Megatron分片或模型并行）：** *激活值沿D ($d_\text{model}$) 分片，参数沿F ($d_{ff}$) 分片。每个块前后对激活值进行AllGather和ReduceScatter。可与FSDP兼容。*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. 流水线并行：** *权重沿层维度分片，激活值沿层维度进行微批次滚动。流水线阶段之间的通信最小（仅跨一跳传输激活值）。为简化表示：*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D][i]$$

### 数据并行

**语法：** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

当您的模型即使只有很小的批处理（>240个标记，以达到计算受限状态）也能装入单个芯片时，**应始终使用简单的数据并行。** 纯数据并行将我们的激活值分布在任意数量的TPU上，只要TPU数量小于我们的批处理大小。前向传播不涉及通信，但在每个步骤结束时，**每个TPU对其本地梯度执行AllReduce操作以在更新参数前同步它们。**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>图示：</b>纯数据并行（前向传播）示意图。我们的激活值（左侧）沿批处理维度完全分片，而权重完全复制，因此每个TPU拥有权重的相同副本。这意味着权重的总内存增加了N倍，但前向传播不需要通信。" %}

{% details 以下是前向和反向传播的完整算法。为简洁起见，我们将dL/dOut记作dOut。 %}

<div markdown=1 class="algorithm">

**纯数据并行算法：**

**前向传播：** 需要计算 Loss[B<sub>X</sub>]

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**反向传播：** 需要计算 dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*不在关键路径上，可异步执行*)
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) (*不在关键路径上，可异步执行*)
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*用于前一*)

</div>

我们忽略损失函数的细节，并将 $\text{Tmp} = W_\text{in} \cdot \text{In}$ 缩写为 $\text{Tmp}$。请注意，虽然我们的最终损失是平均值 **AllReduce**(Loss[B<sub>X</sub>])，但我们只需要在反向传播中对权重梯度进行平均时才需要计算AllReduce。

{% enddetails %}

请注意，前向传播没有通信——**所有通信都在反向传播中！** 反向传播还有一个很好的特性，即AllReduces不在“关键路径”上，这意味着每个AllReduce可以在方便时执行，不会阻塞您执行后续操作。如果通信总成本超过我们的总计算成本，它_仍然可能成为瓶颈_，但从实现角度来看要宽容得多。我们将看到模型/张量并行不具备此特性。

**为什么这样做？** 纯数据并行通过沿批处理维度分割激活值来减少激活内存压力，使我们能够几乎任意地增加批处理大小，只要我们有更多芯片来分割批处理维度。特别是在训练期间，当激活值通常主导我们的内存使用时，这非常有帮助。

**为什么不这样做？** 纯数据并行不会减少模型参数或优化器状态的内存压力，这意味着纯数据并行对于大规模中有趣的模型通常无用，在这些模型中，我们的参数+优化器状态无法装入单个TPU。为了感受规模，如果我们以bf16格式存储参数，以fp32格式存储优化器状态并使用Adam优化器<d-footnote>Adam存储参数、一阶和二阶累加器。由于参数为bfloat16，优化器状态为float32，这给了我们每个参数 `2 + 8 = 10` 字节。</d-footnote>，我们可以容纳的最大模型参数量为 $$\text{TPU 内存} / 10$$，因此例如在具有96GB HBM的TPUv5p芯片上使用纯数据并行，这大约是90亿参数。

<p markdown=1 class="takeaway">**要点**：使用Adam和纯数据并行，我们能训练的最大模型具有 $$\text{num_params} = \text{每设备 HBM} / 10$$。对于TPU v5p，这大约是90亿参数。<d-footnote>请注意，这不包括梯度检查点，因此实际上并不实用。这是一个绝对下限，批处理大小为1个标记。</d-footnote></p>

*为了使这对训练期间的实际模型有用，我们将至少需要部分分片模型参数或优化器。*

**我们何时会因通信而成为瓶颈？** 如上所述，我们每层有两个AllReduce，每个大小为 $$2DF$$（对于bf16权重）。数据并行何时会使我们受限于通信？

如上表所示，设 $C$ = 每芯片FLOPs，$W_{\text{ici}}$ = **双向**网络带宽，$X$ = 批处理分区的分片数量<d-footnote>我们假设此分区在ICI网格上完成，因此相关网络带宽为 $W_\text{ici}$</d-footnote>。 让我们计算执行相关矩阵乘法所需的时间 $$T_\text{math}$$ 和所需的通信时间 $$T_\text{comms}$$。 由于此并行方案在前向传播中不需要通信，我们只需为反向传播计算这些量。

*通信时间：* 从前面的章节我们知道，在1D网格中执行AllReduce所需的时间仅取决于被AllReduce的数组的总字节数和ICI带宽 $W_\text{ici}$；具体来说，AllReduce时间为 $2 \cdot \text{总字节数} / W_\text{ici}$。由于我们需要对 $W_\text{in}$ 和 $W_\text{out}$ 进行AllReduce，因此我们每层有2个AllReduce。 每个AllReduce针对一个权重矩阵，即一个 $DF$ 参数的数组，或 $2DF$ 字节。将所有这些结合起来，单层AllReduce的总时间为

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*矩阵乘法时间：* 每层包含前向传播中的两个矩阵乘法，或反向传播中的四个矩阵乘法，每个需要 $2(B/X)DF$ FLOPs。因此，对于反向传播中的单层，我们有

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

由于我们重叠执行，每层的总时间是这两个量的最大值：

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

当 $$T_\text{math}/T_\text{comms} > 1$$ 时，我们变为计算受限，即当

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

结论是，为了在数据并行下保持计算受限，我们需要每设备批处理大小 $$B / X$$ 超过ICI操作强度 $C / W_\text{ici}$。这最终是以下事实的结果：计算时间随每设备批处理大小而增加，而通信时间与此量无关（因为我们正在传输模型权重）。请注意 $B/X > C/W_\text{ici}$ 条件与单设备计算受限规则 $B > 240$ 的相似性；在那种情况下，规则也
## Takeaways from LLM Training on TPUs
* 提高并行度或减小批次大小（batch size）都会加剧通信瓶颈（communication-bound），因为这些操作减少了每个芯片的计算量。

* 在合理的上下文长度（约32k）内，我们可以将Transformer建模为MLP模块的堆栈，并通过各并行方案如何分片每层中的两到三个主要矩阵乘法（matmul）来定义它们。

* 训练时主要考虑4种并行方案，每种方案都有其带宽和计算需求（数据并行、FSDP、张量并行以及混合FSDP+张量并行）。

| **策略** | **描述** |
| --- | --- |
| **数据并行 (Data Parallelism)** | 激活值按批次分片，其余参数完全复制，反向传播时通过全局归约同步梯度。 |
| **FSDP** | 激活值、权重和优化器状态均按批次分片，权重在使用前汇集，梯度通过归约-分散通信传递。 |
| **张量并行 (Tensor Parallelism, 又称Megatron并行、模型并行)** | 激活值沿$$d_\text{model}$$维度分片，权重沿$$d_{ff}$$维度分片，在W<sub>in</sub>计算前汇集激活值，在W<sub>out</sub>计算后对结果进行归约-分散。 |
| **混合FSDP+张量并行** | 结合上述两种方案，由FSDP汇集经模型分片的权重。 |

各策略对应的"公式"如下：

$$\small
\begin{array}{cc}
\text{策略} & \text{公式}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{TP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{TP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

* 每种策略都存在网络/通信瓶颈临界点，这取决于其单设备计算量与通信量。以下是各策略单层计算与通信量（假设$$X$$为FSDP并行度，$$Y$$为张量并行度）：

$$
\small
\begin{array}{ccc}
\text{策略} & \text{单层计算量} & \text{单层通信量} \\
& \text{(忽略门控求和)} & \text{(字节，前向+反向传播)}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{TP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + TP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

* 纯数据并行很少适用，因为模型及其优化器状态占用的内存约为参数量的10倍。这意味着通常只能在内存中容纳数十亿参数。

* 当$$\text{每分片批次大小} < C / W$$（网络算术强度）时，数据并行和FSDP会达到通信瓶颈。对于ICI网络，该阈值为2550；对于DCN网络，约为71000。增加并行轴数可提升此阈值。

* 当$$\lvert Y\rvert > F / 2550$$时，张量并行达到通信瓶颈。**对于大多数模型，临界点在8-16路并行。** 该限制与批次大小无关。

* 混合FSDP+张量并行可将批次大小降至$$2550^2 / 2F \approx 100$$，这是极低的阈值。

* 跨Pod的数据并行需保证每个Pod至少约71000的批次大小，否则会受限于DCN网络带宽。

* 基本上，若批次足够大或模型足够小，情况就很简单：可采用数据并行或跨DCN的FSDP+数据并行。中间状态才是需要精细优化的场景。
## Some Problems to Work
我们以 LLaMA-2 13B 作为本节的基础模型。以下是模型细节：

| 超参数 (hyperparam) | 值     |
| ------------------- | ------ |
| L                   | 40     |
| D                   | 5,120  |
| F                   | 13824  |
| N                   | 40     |
| K                   | 40     |
| H                   | 128    |
| V                   | 32,000 |

LLaMA-2 具有独立的嵌入矩阵和输出矩阵，以及一个门控 MLP 块。

**问题 1：** LLaMA-2 13B 有多少个参数（我知道这很简单，但算一下）？*请注意，正如在[Transformer 数学](../transformers)中一样，LLaMA-3 有 3 个大的 FFW 矩阵，其中两个是上投影，一个是下投影。在本节中，我们忽略了两个“门控”einsum 矩阵，但它们的行为与本节中的 W<sub>in</sub> 相同。*

{% details 点击这里查看答案。 %}

* FFW 参数：$$3LDF$$ = `8.5e9`
* 注意力参数：$$4DNHL$$ = `4.2e9`
* 词汇表参数：$$2VD$$ = `0.33e9`
* 总计：`8.5e9 + 4.2e9 + 0.33e9 = 13.0e9`，符合预期！

{% enddetails %}

**问题 2：** 假设我们以 BS=16M 个 token 进行训练并使用 Adam。暂时忽略并行性，模型参数、优化器状态和激活值总共使用多少内存？*假设我们以 bf16 格式存储参数，以 fp32 格式存储优化器状态，并且在每层（在三次大型矩阵乘法之后）检查点激活值三次。*

{% details 点击这里查看答案。 %}

用于参数（bf16）和两个优化器状态（fp32，一阶和二阶矩累积器）的总内存为 `(2 + 4 + 4) * 13e9 ~ 130GB`。前两次矩阵乘法后的激活值形状为 $BF$，最后一次后的形状为 $BD$（根据上面的 Transformer 图），因此 bf16 的总内存为 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$，即 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`，其中 `B=16e6`。所有其他激活值基本可以忽略不计。

{% enddetails %}

**问题 3：** 假设我们希望在 TPUv5p 16x16x16 切片上以 32k 序列长度和 3M 个 token 的总批次大小进行训练。假设我们希望使用 bfloat16 权重和 float32 优化器，如上所述。

1. 我们可以使用纯数据并行吗？为什么可以或为什么不行？
2. 我们可以使用纯 FSDP 吗？为什么可以或为什么不行？使用纯 FSDP 时，每个设备将使用多少内存（假设我们仅在三个大型 FFW 矩阵之后进行梯度检查点）？
3. 我们可以使用混合 FSDP + 张量并行吗？为什么可以或为什么不行？如果可以，$X$ 和 $Y$ 应该是什么值？每个设备将存储多少内存？仅使用基于屋顶线模型的 FLOPS 估计并忽略注意力机制，在 40% 的 MFU 下，每个训练步骤需要多长时间？

{% details 点击这里查看答案。 %}

首先，列出一些数字。使用 32k 的序列长度和 3M 的批次大小，我们的序列批次大小为 96。在 TPU v5p 16x16x16 切片上，我们有 `393TB` 的 HBM。

1. 我们无法使用纯数据并行，因为它会在每个芯片上复制参数和优化器状态，这些已经大约占用 130GB（来自问题 2），这比我们每片芯片的 HBM（96GB）还要多。

2. 让我们先纯粹看内存。将问题 2 中的 BS=16M 替换为 3M，我们得到 `~7.86e12` 的总检查点激活值，加上 1.3e11 的优化器状态，总计几乎正好是 8e12 = 8TB。TPUv5p 切片总共有 `393TB` 的 HBM，因此我们安全地低于 HBM 限制。接下来，让我们看看是通信受限还是计算受限。使用 4096 个芯片和 3 个并行轴，我们可以实现的最小批次大小为 `850 * 4096 = 3.48M` 个 token。这略高于我们 3M 的批次大小。因此我们实际上是通信受限的，这很糟糕。所以总体答案是 **不，我们无法单独使用 FSDP**。

3. 现在我们知道主要问题是通信受限，所以让我们代入一些数字。首先，根据上文，我们知道使用混合 FSDP + 张量并行时，每芯片批次大小需要大于 $2550^2 / 2F = 235$。这意味着我们理论上可以这样做！让我们确定每种的分配。

我们有规则 $X_{opt} = \sqrt{(B / F) \cdot (M_X / M_Y) \cdot N}$，所以这里我们有 `sqrt(3e6 * 2 * 4096 / 13824) = 1333`，这意味着我们将进行大约 1024 路数据并行和 4 路张量并行。每片 TPU 的内存将与 (2) 中相同，步骤时间将是 `6 * 3e6 * 13e9 / (4096 * 4.6e14 * 0.4) = 300ms`。

{% enddetails %}

<h3 markdown=1 class="next-section">第 5 部分到此结束！第 6 部分将把这些内容应用于真实的 LLaMA 模型，[请点击这里](../applied-training)！</h3>
## Appendix
### 附录 A：推导反向传播通信

前文我们将 Transformer 层的前向传播简化为 Out[B, D] = In[B, D] *<sub>D</sub> W<sub>in</sub>[D, F] *<sub>F</sub> W<sub>out</sub>[F, D]。如何推导反向传播所需的通信？

这可以自然地从上一节中针对单次矩阵乘法 **Y = X * A** 的规则推导而来：

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

应用此规则，我们得到以下公式（令 Tmp[B, F] 表示 In[B, D] * W<sub>in</sub>[D, F]）：

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] *<sub>B</sub> dOut[B, D]
2. dTmp[B, F] = dOut[B, D] *<sub>D</sub> W<sub>out</sub>[F, D]
3. dW<sub>in</sub>[D, F] = In[B, D] *<sub>B</sub> dTmp[B, F]
4. dIn[B, D] = dTmp[B, F] *<sub>F</sub> W<sub>in</sub>[D, F]

</div>

请注意，这些公式是数学表述，未涉及分片（sharding）。反向传播的任务是计算这四个量。因此，要确定所需的通信，我们只需获取上述四个方程中所有待进行矩阵乘法的量（Tmp、dOut、W<sub>out</sub>、W<sub>in</sub>）的分片方式——这些由我们的并行化方案（parallelization scheme）指定——然后运用分片矩阵乘法（sharded matmuls）的规则来推断需要执行哪些通信。注意，dOut 的分片方式与 Out 相同。