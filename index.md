---
layout: distill
title: "How to Scale Your Model"
subtitle: "A Systems View of LLMs on TPUs"
# permalink: /main/
description: "Training LLMs often feels like alchemy, but understanding and optimizing the performance of your models doesn't have to. This book aims to demystify the science of scaling language models: how TPUs (and GPUs) work and how they communicate with each other, how LLMs run on real hardware, and how to parallelize your models during training and inference so they run efficiently at massive scale. If you've ever wondered \"how expensive should this LLM be to train\" or \"how much memory do I need to serve this model myself\" or \"what's an AllGather\", we hope this will be useful to you."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "Part 0: Intro"

next_section_url: roofline
next_section_name: "Part 1: Rooflines"

bibliography: main.bib

citation: true

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
  - name: High-Level Outline
  - name: Links to Sections

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
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

深度学习在很大程度上仍像一种黑魔法，但优化模型性能却不必如此——即使在巨大规模下也是如此！相对简单的原理普遍适用——从处理单个加速器到数万个加速器——理解这些原理能让你做许多有用的事情：

- 估算模型各部分与其理论最优值的接近程度。
- 在不同规模下对不同的并行化方案做出明智选择（如何将计算分配到多个设备上）。
- 估算训练和运行大型Transformer模型所需的成本和时间。
- 设计能利用[特定](https://arxiv.org/abs/2205.14135)[硬件](https://arxiv.org/abs/1911.02150)[特性](https://arxiv.org/abs/2007.00072)的算法。
- 设计基于对当前算法性能限制的清晰理解而驱动的硬件。

**预期背景：** 我们将假定你对大语言模型（LLM）和Transformer架构有基本了解，但不一定了解它们如何在大规模下运行。你应该了解LLM训练的基础知识，最好对JAX有基本熟悉。一些有用的背景阅读材料可能包括[这篇博客文章](https://jalammar.github.io/illustrated-transformer/)中对Transformer架构的说明和[原始Transformer论文](https://arxiv.org/abs/1706.03762)。另请查看[此列表](conclusion#further-reading)以获取更多有用的当前及未来阅读材料。

**目标与反馈：** 学完后，你应该能够自信地估算在给定硬件平台上Transformer模型的最佳并行化方案，并大致了解训练和推理所需的时间。如果不能，请给我们发邮件或留言！我们很乐意了解如何能讲解得更清楚。

<p markdown=1 class="announce">你可能也会喜欢阅读关于NVIDIA GPU的新增[第12节](gpus)！</p>

### 为什么你应该关心这些？

三、四年前，我认为大多数机器学习（ML）研究人员并不需要理解本书中的任何内容。但今天，即使是"小型"模型的运行也如此接近硬件极限，以至于进行新颖研究需要你思考大规模下的效率。<d-footnote>历史上，机器学习研究在系统创新和软件改进之间遵循着一种类似钟摆的循环。Alex Krizhevsky不得不编写复杂的CUDA代码来让CNN运行得更快，但在几年内，像Theano和TensorFlow这样的库意味着你不必再这样做。也许这种情况也会在这里发生，几年后本书中的所有内容都会被抽象掉。但扩展定律已经将我们的模型永久推向硬件的最前沿，在可预见的未来，进行前沿研究似乎将不可避免地与理解如何高效地将模型扩展到大型硬件拓扑结构相关联。</d-footnote> **如果以牺牲20%的屋顶线效率为代价换来基准测试上20%的提升，那是毫无意义的。** 有前景的模型架构常常失败，要么因为它们_无法_高效地大规模运行，要么因为没有人投入精力使其能够做到。

**"模型扩展"的目标是能够增加用于训练或推理的芯片数量，同时实现吞吐量成比例、线性的增长。** 这被称为"*强扩展*"。尽管增加额外的芯片（"并行化"）通常会减少计算时间，但它也带来了芯片间额外通信的开销。当通信时间超过计算时间时，我们就变得"受通信限制"，无法实现强扩展。<d-footnote>随着计算时间减少，你通常也会面临单芯片级别的瓶颈。你崭新的TPU或GPU可能标称能每秒执行500万亿次运算，但如果不小心，如果它因在内存中搬运参数而陷入困境，它同样可能只能发挥十分之一的性能。单芯片计算、内存带宽和总内存之间的相互作用对于扩展故事至关重要。</d-footnote> 如果我们足够了解我们的硬件，就能预判这些瓶颈将在哪里出现，从而设计或重新配置模型以避免它们。<d-footnote>硬件设计者面临相反的问题：构建硬件时需要提供恰好足够的计算能力、带宽和内存，同时最小化成本。你可以想象这种"协同设计"问题的压力有多大：你必须赌定当第一批芯片实际可用时（通常是2到3年后）算法会是什么样子。TPU的故事在这个游戏中是一个巨大的成功。矩阵乘法是一个独特的算法，因为它使用的每字节内存浮点运算次数（N FLOPs per byte）远多于几乎所有其他算法，早期的TPU及其脉动阵列架构在它们诞生之时实现了比GPU好得多的性能/成本比。TPU是为机器学习工作负载设计的，而带有张量核心的GPU也在迅速改变以填补这一领域。但你可以想象，如果神经网络没有兴起，或者发生了TPU（本质上不如GPU灵活）无法处理的根本性变化，那将会多么昂贵。</d-footnote>

*本书的目标是解释TPU（和GPU）硬件如何工作，以及Transformer架构如何演进以在当前硬件上表现良好。我们希望这对设计新架构的研究人员和致力于让当前一代大语言模型快速运行的工程师都有用。*
## High-Level Outline
本书整体结构如下：

[第1节](roofline)阐述屋顶线（roofline）分析及其限制扩展能力的因素（通信、计算与内存）。[第2节](tpus)与[第3节](sharding)将深入探讨TPU的工作原理——既包括单芯片运行机制，更关键的是作为通过带宽与延迟受限的芯片间链路互联的系统如何运作。我们将解答以下问题：

* 特定规模的矩阵乘法耗时多久？在何种情况下会受计算、内存或通信带宽限制？
* TPU如何连接构成训练集群？系统各部分带宽是多少？
* 在多个TPU间执行数组的聚集（gather）、分散（scatter）或重分布需要多久？
* 如何高效完成在不同设备间分布差异的矩阵乘法？

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>图示：</b><a href='tpus'>第2节</a>中的示意图展示TPU执行逐元素乘法的过程。根据数组规模与链路带宽差异，可能出现计算受限（完全使用硬件算力）或内存受限（受内存加载瓶颈制约）的情况。" %}

五年前机器学习领域存在多样化架构——卷积网络（ConvNets）、长短期记忆网络（LSTMs）、多层感知机（MLPs）、Transformer——而如今我们基本只剩Transformer架构<d-cite key="transformers"></d-cite>。我们坚信有必要理解Transformer架构的每个组件：每层矩阵的精确尺寸、归一化（normalization）发生位置、各部分参数量与浮点运算次数<d-footnote>FLoating point OPs（浮点运算次数），基本等同于所需的加法和乘法总次数。尽管许多资料将FLOPs理解为"每秒运算次数"，我们在此明确使用FLOPs/s表示该概念。</d-footnote>。[第4节](transformers)将细致解析这套"Transformer数学"，展示如何计算训练与推理阶段的参数量及浮点运算次数。这能帮助我们了解模型内存占用、计算或通信耗时，以及注意力机制何时会比前馈网络块更关键。

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>图示：</b>标准Transformer层示意图，每个矩阵乘法（matmul）以圆圈内点表示。所有参数（归一化参数除外）均标注为紫色。<a href='transformers'>第4节</a>将对该图进行详细解析。" %}

[第5节：训练](training)与[第7节：推理](inference)构成本书核心，我们将探讨根本问题：给定特定规模模型和芯片数量，如何对模型进行并行化（parallelize）以保持"强扩展（strong scaling）"状态？这个问题看似简单，答案却异常复杂。从宏观看，主要存在四种用于跨芯片分割模型的并行化技术（**数据并行**、**张量并行**、**流水线并行**与**专家并行**），以及多种降低内存需求的技术（**重计算（rematerialization）**、**优化器/模型分片（即ZeRO技术）**、**主机卸载（host offload）**、**梯度累积（gradient accumulation）**）。本书将讨论其中多项技术。

我们希望通过这些章节的学习，您能自主为新架构或新场景选择合适方案。[第6节](applied-training)与[第8节](applied-inference)是实践教程，将上述概念应用于主流开源模型LLaMA 3。

最后，[第9节](profiling)与[第10节](jax-stuff)探讨如何在JAX中实现部分技术方案，以及出现问题时如何进行代码性能分析（profiling）与调试。[第12节](gpus)是新增章节，将深入解析GPU工作原理。

全书配有实践习题，建议您按需选读各节内容，无需强求顺序阅读。欢迎随时反馈意见。当前版本为草案，将持续修订完善。感谢支持！

*特别鸣谢James Bradbury与Blake Hechtman对本书核心思想的贡献。*

<h3 markdown=1 class="next-section">闲言少叙，[点击这里进入第1节](roofline)了解TPU屋顶线分析。</h3>
## Links to Sections
*本系列篇幅可能超出必要长度，但我们希望这不会让您望而却步。前三章是预备知识，若您已熟悉相关内容可选择跳过，不过它们介绍了后续使用的符号体系。最后三部分可能最具实践价值，因为它们解释了如何处理真实模型。*

**第一部分：预备知识**

* [**第1章：Roofline分析简述**](roofline)。算法受三方面限制：计算、通信和存储。我们可以通过这些因素近似评估算法的运行速度。

* [**第2章：理解TPU的工作原理**](tpus)。TPU如何运作？这对可训练和可部署的模型有何影响？

* [**第3章：分片矩阵及其乘法运算**](sharding)。本章将通过我们最常用的操作——（分片）矩阵乘法，阐释模型分片与多TPU并行计算。

**第二部分：Transformer架构**

* [**第4章：Transformer必备数学知识**](transformers)。Transformer在前向传播和反向传播中消耗多少FLOPs？如何计算参数量？KV缓存的规模如何？本章将逐步推导这些计算。

* [**第5章：Transformer训练并行化策略**](training)。完全分片数据并行（FSDP）。Megatron分片。流水线并行。给定特定数量的芯片，如何以最高效率使用指定批量大小训练特定规模的模型？

* [**第6章：在TPU上训练LLaMA 3**](applied-training)。如何在TPU上训练LLaMA 3？需要多长时间？成本如何？

* [**第7章：Transformer推理全解析**](inference)。模型训练完成后需要进行部署。推理场景引入了新的考量因素——延迟，并改变了内存使用格局。我们将讨论分离式服务架构原理以及KV缓存的优化思路。

* [**第8章：在TPU上部署LLaMA 3**](applied-inference)。在TPU v5e上部署LLaMA 3的成本是多少？如何权衡延迟与吞吐量？

**第三部分：实践教程**

* [**第9章：TPU代码性能分析**](profiling)。真实大语言模型从不像上述理论那样简单。本章将解析JAX+XLA技术栈，并介绍如何使用JAX/TensorBoard性能分析器调试实际问题。

* [**第10章：使用JAX编程TPU**](jax-stuff)。JAX提供了一系列神奇的并行计算API，但您需要掌握其使用方法。本章包含趣味实例与详尽解析。

**第四部分：总结与补充内容**

* [**第11章：总结与延伸阅读**](conclusion)。关于TPU与大语言模型的结语及进阶阅读材料。

* [**第12章：理解GPU的工作原理**](gpus)。关于GPU的补充章节：工作原理、互联方式及其与TPU在Roofline模型上的差异对比。